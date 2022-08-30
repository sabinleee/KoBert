from modelinit import *

def bert_base(dataset_train, dataset_test):
    bertinit = ModelInit('bert-base-multilingual-cased')
    
    tok = bertinit.tokenizer.tokenize
    vocab = bertinit.vocab
    train_batch_size = bertinit.train_batch_size
    test_batch_size = bertinit.test_batch_size

    data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, 64, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, 64, True, False)

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
    
    bertinit.train_dataloader = train_dataloader
    bertinit.test_dataloader = test_dataloader
    
    model = BERTClassifier(bertinit.model,  dr_rate=0.5).to(device)
    bertinit.define_hyperparameters(model, train_dataloader, epochs=5)

    return model, bertinit

def kobert(dataset_train, dataset_test):
    kobertinit = ModelInit('skt/kobert-base-v1')

    tok = kobertinit.tokenizer.tokenize
    vocab = kobertinit.vocab
    train_batch_size = kobertinit.train_batch_size
    test_batch_size = kobertinit.test_batch_size

    data_train = BERTDataset(dataset_train, 0, 1, tok, vocab, 64, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, vocab, 64, True, False)

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
    for e in range(epochs):
        print("")
        print('======== Epoch {:} / {:} ========'.format(e + 1, epochs))
        print('Training...')
        
        t0 = time.time()
        
        total_loss = 0
        train_acc = 0.0
        test_acc = 0.0

        model.train()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
            # progress bar
            if batch_id % 500 == 0 and not batch_id == 0:
                elapsed = format_time(time.time() - t0)
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(batch_id, len(train_dataloader), elapsed))

            modelinit.optimizer.zero_grad()

            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)
            
            out = model(token_ids, valid_length, segment_ids)

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

            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length= valid_length
            label = label.long().to(device)

            out = model(token_ids, valid_length, segment_ids)
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
    dataset_train = nlp.data.TSVDataset("ratings_train.txt", field_indices=[1,2], num_discard_samples=1)
    dataset_test = nlp.data.TSVDataset("ratings_test.txt", field_indices=[1,2], num_discard_samples=1)
    
    # ---------------------------------------------------------------------- #
    #                      bert base multilingual cased                      #
    # ---------------------------------------------------------------------- #
    
    # bert_model, bertinit = bert_base(dataset_train, dataset_test)
    # train_eval(bert_model, bertinit)
    # torch.save(model.state_dict(), './model/bert_base_classifier.pt')
    # torch.save(model, './model/bert_base_model.bin')   
    
    
    # ---------------------------------------------------------------------- #
    #                                kobert                                  #
    # ---------------------------------------------------------------------- #

    kobert_model, kobertinit = kobert(dataset_train, dataset_test)
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