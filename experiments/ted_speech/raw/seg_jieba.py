# encoding=utf-8
import jieba
import sys
import os
import glob

#input_folder = sys.argv[1]
#output_folder = sys.argv[2]

list_of_files = glob.glob('/home/cxz142930/ByteNet/asset/ted/text.cn/*.txt')
for fileName in list_of_files:
    f_cn = open(fileName,'r')
    seg_text = open(fileName+"_seg",'w')

    for line in f_cn:
        line = unicode(line.strip(), 'utf-8')
        #line = bytes(line.strip(),'utf-8')
        seg_list = jieba.cut(line, cut_all=False)
        new_line=' '.join(seg_list)
        seg_text.write((new_line+u'\n').encode('utf-8'))


    f_cn.close()
    seg_text.close()

#seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
#print("Full Mode: " + "/ ".join(seg_list))  # 全模式

#seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
#print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

#seg_list = jieba.cut("我的儿子在2001年9月11日早晨")  # 默认是精确模式
#print(", ".join(seg_list))

#seg_list = jieba.cut_for_search("小明硕士Sunglass毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
#print(", ".join(seg_list))
