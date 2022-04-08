import jieba
import math
import os
import re

#本代码整体架构参考了CSDN上的文章《深度学习与自然语言处理实验——中文信息熵的计算》，同时增加了一些自己的理解
#在之前并未做过类似大批量处理文件的操作，通过学习参考别人的代码做出尝试

#首先，初始化做一个类
class pre_process():

    def __init__(self, fileroute):
        self.fileroute = fileroute

    def giv_process(self):
        return pre_process.getCorp(self, self.fileroute)
 
    def getCorp(self, fileroute):     #定义一个函数，主要目的是处理数据库，包括删除标点，多余字等等工作，得到语料库，为后面的计算提供条件
        #这里定义几个必要量
        rdelate = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:：;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'  # 定义了要删掉的一些符号，保证文章中只剩下中文，便于后续处理
        allfile_list = os.listdir(fileroute)    #列出路径下所有文件，储存在allfile_list当中，后面使用
        corpus = []   #定义一个空的数组，将未来的语料库放入进去
        count=0   #定义一个计数器用于计算文字数量，便于平均数字量的计算

        for file in allfile_list:   #将小说一本一本提取出来，每一本分别处理 
            path  = os.path.join(fileroute, file)   #该函数用来整合路径，输出即为当前文件的路径
            if os.path.isfile(path):  #做一个判断，只有路径正确才进行下一步处理，否则进入下一个路径  
                with open(os.path.abspath(path), "r", encoding='ansi') as file:

                    file_read = file.read()    #导入文件
                    file_read = re.sub(rdelate, '', file_read)   #把多余的符号删除，符号上面有过定义
                    file_read = file_read.replace("本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com", '')    #本身缺少这句话，打开文件后发现该段话与正文无关，故删除
                    file_read = file_read.replace("\n", '') #下面两句话删除一些空格和回车，让整个语料库连接起来，方便操作
                    file_read = file_read.replace(" ", '')
                    #下面两句话分别是记录语料库字数和修改过后的语料库，也是我们这个函数整个处理的最终需要得到的结果，

                    count += len(file_read)     #通过循环记录所有文章总字数
                    corpus.append(file_read)    #通过循环将所有文章导入数组，形成语料库
        return corpus,count


#做一个词频统计函数，三种ngram模型计算方式不同，方便后面计算，这里设计了三个函数，没有采用for的方式做成一个，方便阅读
#同时借鉴了csdn上的计算方法
def get_fsigram_tf(tf_dic, words):   #一元模型词频统计

    for i in range(len(words)-1):
        tf_dic[words[i]] = tf_dic.get(words[i], 0) + 1

def get_secgram_tf(tf_dic, words):   #二元模型词频统计
    for i in range(len(words)-1):
        tf_dic[(words[i], words[i+1])] = tf_dic.get((words[i], words[i+1]), 0) + 1

def get_trigram_tf(tf_dic, words):   #三元模型词频统计
    for i in range(len(words)-2):
        tf_dic[((words[i], words[i+1]), words[i+2])] = tf_dic.get(((words[i], words[i+1]), words[i+2]), 0) + 1

#下面开始做三个函数，分别使用一元，二元，三元模型
def cal_unigram(corpus,count):
    all_words = []      #定义新数组，用于储存分词
    words_count = 0     #记录分词个数，用于后续计算
    words_tf = {}       #用于储存词频函数计算出的词频
    information_entropy = []    #将每本小说的信息熵输入其中
    for word in corpus: #循环调用整个库，每次计算一本小说
        for x in jieba.cut(word):   #采用jieba做分词，将改本小说分解
            all_words.append(x)     #将分词接到all_words
            words_count += 1        #每有一个分词即进一次循环，得到总的分词个数
        get_fsigram_tf(words_tf, all_words)     #通过词频函数计算出词频，后面计算使用
        all_words = []                          #在一本小说计算完成后重置，准备用于计算下一本小说
    print("字数:", count)                       #输出语料库字数，该值在前面已经计算
    print("平均词长:", count / words_count)   #输出平均字长
    print("分词数:", words_count)               #输出分次数 
    #计算信息熵，利用公式，可以得到每一本小说的信息熵
    for fri_word in words_tf.items():
        information_entropy.append(-(fri_word[1] / words_count) * math.log(fri_word[1] / words_count, 2))  #计算一元信息熵公式
    print("一元模型----平均信息熵:", sum(information_entropy), "比特/词")   #输出结果，算一个总值

def cal_bigram(corpus, count):
    all_words = []
    words_count = 0
    words_tf = {}
    secgram_tf = {}
    information_entropy = []  
    for word in corpus:
        for x in jieba.cut(word):
            all_words.append(x)
            words_count += 1
        get_fsigram_tf(words_tf, all_words)
        get_secgram_tf(secgram_tf, all_words)
        all_words = []
    print("字数:", count)
    print("平均词长:", count / words_count)
    print("分词数:", words_count)

    secgram_len = sum([dic[1] for dic in secgram_tf.items()])
    print("二元模型长度:", secgram_len)
#这里计算使用公式，同时参考网上代码
    for sec_word in secgram_tf.items():
        jp_xy = sec_word[1] / secgram_len                        # 计算联合概率
        cp_xy = sec_word[1] / words_tf[sec_word[0][0]]           # 计算条件概率
        information_entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算二元模型的信息熵
    print("二元模型----平均信息熵:", sum(information_entropy), "比特/词")



def cal_trigram(corpus,count):

    all_words = []
    words_count = 0
    words_tf = {}
    trigram_tf = {}
    information_entropy = []  
    for word in corpus:
        for x in jieba.cut(word):
            all_words.append(x)
            words_count += 1
        get_secgram_tf(words_tf, all_words)
        get_trigram_tf(trigram_tf, all_words)
        all_words = []
    print("字数:", count)
    print("平均词长:", round(count / words_count, 5))
    print("分词数:", words_count)
   
    trigram_len = sum([dic[1] for dic in trigram_tf.items()])
    print("三元模型长度:", trigram_len)
#计算信息熵
    for tri_word in trigram_tf.items():
        jp_xy = tri_word[1] / trigram_len                        # 计算联合概率
        cp_xy = tri_word[1] / words_tf[tri_word[0][0]]           # 计算条件概率
        information_entropy.append(-jp_xy * math.log(cp_xy, 2))  # 计算三元模型的信息熵

    print("三元模型----平均信息熵::", sum(information_entropy), "比特/词")


#主函数
if __name__ == '__main__':      
    cons = pre_process("./dataset")     
    corpus,count = cons.giv_process()
    cal_unigram(corpus, count)      #一元模型计算
    cal_bigram(corpus,count)        #二元模型计算
    cal_trigram(corpus,count)       #三元模型计算
    

