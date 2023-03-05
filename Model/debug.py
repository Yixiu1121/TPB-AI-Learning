timeList = [3,3,3,9,8,7,3] #(0,4,8,12,16,23,27)
# del_tuple = ()
# for num in range(0,len(timeList)):
#     del_tuple += tuple([sum(timeList[1:num+1])+num])
# print(del_tuple)
#
insert_tuple = ()
for num in range(0,len(timeList)):
    insert_tuple += tuple([sum(timeList[:num+1])])
print(insert_tuple)
3,6,9,18,26,33,36