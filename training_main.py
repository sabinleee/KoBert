from modelinit import *


def bert(dataset_train, dataset_test, pretrained_model):
    kobertinit = ModelInit(pretrained_model)

    tok = kobertinit.tokenizer
    train_batch_size = kobertinit.train_batch_size
    test_batch_size = kobertinit.test_batch_size

    data_train = BERTDataset(dataset_train, 'document', 'label', tok)
    data_test = BERTDataset(dataset_test, 'document', 'label', tok)
    
    train_dataloader = DataLoader(
                        data_train, 
                        batch_size=train_batch_size, 
                        num_workers=4,
                        shuffle=True,
                        pin_memory=True
                        )
        
    test_dataloader = DataLoader(
                        data_test, 
                        batch_size=test_batch_size, 
                        num_workers=4,
                        shuffle=True,
                        pin_memory=True
                        )
    
    kobertinit.train_dataloader = train_dataloader
    kobertinit.test_dataloader = test_dataloader
        
    model = BERTClassifier(kobertinit.model,  dr_rate=0.5).to(device)
    kobertinit.define_hyperparameters(model, train_dataloader, epochs=5)

    return model, kobertinit


def train_eval(model, modelinit):
    epochs = modelinit.epochs
    train_dataloader = modelinit.train_dataloader
    test_dataloader = modelinit.test_dataloader
    # dataloader가 iterable 하지 않음 왜?
    
    for e in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(e + 1, epochs))
        print('Training...')
        
        t0 = time.time()
        
        total_loss = 0
        train_acc = 0.0
        test_acc = 0.0

        model.train()
        
        for batch_id, batch in enumerate(iter(train_dataloader)):
            # progress bar
            token_ids = batch['input_ids'].long().to(device)
            token_type_ids = batch['token_type_ids'].long().to(device)
            attention_mask = batch['attention_mask'].long().to(device)
            label = batch['label'].long().to(device)
                        
            if batch_id % 500 == 0 and not batch_id == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(batch_id, len(train_dataloader), elapsed))

            modelinit.optimizer.zero_grad()
            
            out = model(token_ids, token_type_ids, attention_mask, modelinit.args.bert_model)
            
            loss = modelinit.loss_fn(out, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            modelinit.optimizer.step()
            modelinit.scheduler.step()

            train_acc += calc_accuracy(out, label)            
                
        print("")
        print("  Average training loss: {0:.2f}".format(loss.data.cpu().numpy()))
        print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        print("  epoch {} / train acc: {}".format(e+1, train_acc / (batch_id+1)))
        
        modelinit.train_loss_per_epoch.append(loss.data.cpu().numpy())
        modelinit.train_accuracy_per_epoch.append(train_acc / (batch_id+1))


        # ========================================
        #               Validation
        # ========================================

        print("")
        print("Running Validation...")
        
        t0 = time.time()
        
        model.eval()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):

            token_ids = batch['input_ids'].long().to(device)
            token_type_ids = batch['token_type_ids'].long().to(device)
            attention_mask = batch['attention_mask'].long().to(device)
            label = batch['label'].long().to(device)

            out = model(token_ids, attention_mask, token_type_ids)
            test_acc += calc_accuracy(out, label)

        print("  epoch {} / test acc: {}".format(e+1, test_acc / (batch_id+1)))
        
        modelinit.accuracy = test_acc / (batch_id+1)
    
    print("")
    print("Training complete!") 
    
    return model


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc 


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# time format
def format_time(elapsed):
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # change to hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

    
def main():
    dataset_train = pd.read_csv("ratings_train.txt", sep="\t").drop(columns=["id"]).dropna(how="any")
    dataset_test = pd.read_csv("ratings_test.txt", sep="\t").drop(columns=["id"]).dropna(how="any")
    
    # ---------------------------------------------------------------------- #
    #                      bert base multilingual cased                      #
    # ---------------------------------------------------------------------- #
    bert_model, bertinit = bert(dataset_train, dataset_test, 'bert-base-multilingual-cased')
    model = train_eval(bert_model, bertinit)
    torch.save(model.state_dict(), './model/bert_base_classifier.pt')
    torch.save(model, './model/bert_base_model.bin')   
    
    
    # ---------------------------------------------------------------------- #
    #                                kobert                                  #
    # ---------------------------------------------------------------------- #
    kobert_model, kobertinit = bert(dataset_train, dataset_test, 'skt/kobert-base-v1')
    model = train_eval(kobert_model, kobertinit)
    
    torch.save(model.state_dict(), './model/kobert_classifier.pt')
    torch.save(model, './model/kobert_model.bin')   

if __name__ == "__main__":
    # set logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.info("Started")
    
    # set seed
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    main()