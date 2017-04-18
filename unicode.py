# -*- coding: utf-8 -*-
# unicode及 str兩種物件，分別對應到 Unicode以及編碼狀態
# http://dannypheobe.blogspot.tw/2017/02/python2-python3.html

### input ###
print type("中文")
print type("中文".decode("utf-8"))
print type(u"中文")
### output ###
#<type 'str'>
#<type 'unicode'>
#<type 'unicode'>

print "中文" # encoded in utf-8
print "中文".decode("utf-8").encode("big5") # encoded in big5

### input ###
print len("中文")
print len("中文".decode("utf-8"))
print len(u"中文")
### output ###
#6
#2
#2
