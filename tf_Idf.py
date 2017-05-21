# -*- coding: utf-8 -*-
'''
Author: razrlele
Email: razrlele@gmail.com
'''
import jieba
import json
from jieba import analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def jieba_tokenize(text):
    return jieba.lcut(text)

# def find_keyword_number(result,tfidf_matrix,jsonvbchart,num_clusters):
#     result=[]
#     for i in num_clusters:
#         for j in len(tfidf_matrix):
#             1
#
#     return 1



tfidf_vectorizer = TfidfVectorizer(tokenizer=jieba_tokenize, \
                                   lowercase=False)

'''
tokenizer: 指定分词函数
lowercase: 在分词之前将所有的文本转换成小写，因为涉及到中文文本处理，
所以最好是False
'''
text_list = ["习近平表示，今年是香港回归祖国20周年。20年来，“一国两制”在香港的实践取得巨大成功。宪法和基本法规定的特别行政区制度有效运行，香港保持繁荣稳定，国际社会给予高度评价。与此同时，作为一项开创性事业，“一国两制”在香港的实践也需要不断探索前进。20年来，香港经历了不少风风雨雨，这个阶段有挑战和风险，也充满机遇和希望。作为新一任行政长官，你责任重大、使命光荣。中央坚持“一国两制”、“港人治港”、高度自治的决心坚定不移，不会变、不动摇，将全力支持你和新一届特别行政区政府依法施政。希望你不负重托，带领特别行政区政府和社会各界，紧紧依靠广大香港同胞，全面准确贯彻落实“一国两制”方针和基本法，团结包容，勠力同心，锐意进取，为香港发展进步作出贡献。", "开放日上，30余位市民参观了人民公墓的业务办理大厅、骨灰安葬告别仪式和怀思阁骨灰永久安放方式。尤其在参观怀思阁时，听说500元可以永久性存放骨灰，而且环境安静优美，有多位市民均询问“可以预定吗？”工作人员表示，“随来随办。”有位80多岁的老人还是不放心，拉着工作人员的手说，“我还是先预定吧，我怕过两年就卖没了。", \
             "据当地警方介绍，１１日晚７时左右，载有球员和教练的多特蒙德队大巴在离开酒店前往西格纳尔·伊杜纳公园球场途中遭遇三次爆炸物袭击，大巴挡风玻璃被震碎，一位名叫马克·巴特拉的球员在爆炸中受伤，并被送往医院。", "目前，人民公墓安放着10万份亡者骨灰，有墓穴近7万个，其中4万老墓已到续租改造的时候，昨天参加开放日的多为老年市民，都很关心如何为逝去已四五十年的父母续租或者迁坟，也有市民咨询是否有新墓穴可以出租？人民公墓工作人员表示，由于公墓建园时间较久，土地资源早就趋于饱和，近些年来，公墓主要业务是以老墓续租改造、合葬、生态节地葬为主。符合条件的市民可选择骨灰墙安葬，其他市民可选择怀思阁存放骨灰。"]
# 需要进行聚类的文本集
tfidf_vectorizer.fit(text_list)
# print json.dumps(tfidf_vectorizer.vocabulary_, ensure_ascii=False, indent=2)
vbchart=dict(map(lambda t:(t[1],t[0]), tfidf_vectorizer.vocabulary_.items()))
jsonvbchart= json.dumps(vbchart, ensure_ascii=False, indent=2)
# print jsonvbchart
tfidf_matrix = tfidf_vectorizer.transform(text_list)
# print tfidf_matrix[2]+tfidf_matrix[3]
num_clusters = 3
km_cluster = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, \
                    init='k-means++', n_jobs=-1)
'''
n_clusters: 指定K的值
max_iter: 对于单次初始值计算的最大迭代次数
n_init: 重新选择初始值的次数
init: 制定初始值选择的算法
n_jobs: 进程个数，为-1的时候是指默认跑满CPU
注意，这个对于单个初始值的计算始终只会使用单进程计算，
并行计算只是针对与不同初始值的计算。比如n_init=10，n_jobs=40,
服务器上面有20个CPU可以开40个进程，最终只会开10个进程
'''
# 返回各自文本的所被分配到的类索引
result = km_cluster.fit_predict(tfidf_matrix)

print "Predicting result: ", result

# find_keyword_number(result,tfidf_matrix,jsonvbchart,num_clusters)
keywordlist=[]
for i in range(num_clusters):
    text=""
    for j in range(len(result)):
        if result[j]==i:
            text=text+text_list[j]

    keywords = analyse.textrank(text)
    keywordlist.append(keywords[0])
jsonresult={
        "type": "force",
        "categories": [
            {
                "name": "HTMLElement",
                "keyword": {},
                "base": "HTMLElement"
            },
            {
                "name": "WebGL",
                "keyword": {},
                "base": "WebGLRenderingContext"
            },
            {
                "name": "SVG",
                "keyword": {},
                "base": "SVGElement"
            }
        ],
        "nodes": [
            {
                "name": "AnalyserNode",
                "value": 2,
                "category": 1
            },
            {
                "name": "Analyserb",
                "value": 1,
                "category": 2
            }
        ],
        "links": [
            {
                "source": 0,
                "target": 1
            },
            {
                "source": 1,
                "target": 2
            }
        ]
    }

print jsonresult
print json.dumps(keywordlist, ensure_ascii=False, indent=2)