import requests as req
from bs4 import BeautifulSoup as bs


class HuggingFaceModelCrawl():
    '''
    허깅페이스 모델 정보 출력
    '''
    def __init__(self) -> None:
        pass

    def get_title(self, page = None , sort_order=None):
        if page is None:
            page = 1
        if sort_order is None:
            sort_order = 'downloads'
        url = 'https://huggingface.co/models?language=ko&p={}&sort={}'.format(page, sort_order)
        res = req.get(url)
        soup = bs(res.text, 'html.parser')

        model_list = []
        [model_list.append(model.text) for model in soup.select('a > header > h4')]
            
        return model_list
    
    def get_sort_order(self):
        sort_order = ['downloads', 'modified', 'likes']
        return sort_order


if __name__ == '__main__':
    hf = HuggingFaceModelCrawl()
    print(hf.get_title())
    print(hf.get_sort_order())