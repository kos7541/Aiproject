import chardet
rawdata = open('C:\\milvus\\2019백서.csv', 'rb').read()
result = chardet.detect(rawdata)
print(result)