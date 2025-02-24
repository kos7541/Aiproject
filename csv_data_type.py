import chardet
rawdata = open('C:\\milvus\\test1.csv', 'rb').read()
result = chardet.detect(rawdata)
print(result)