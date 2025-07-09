liy= ["coin","compass","coral","crystal","diamond","emerald","fossil","key","letter","shell","treasure_box"]
for i in range(8000):
    with open("C:\\Users\\padma\\Downloads\\training_data\\mixed\\"+str(i+32000)+".txt", "r") as file:
        with open("C:\\Users\\padma\\Downloads\\training_data\\txt\\"+str(i+32000)+".txt", "a") as file2:
            for line in file:
                li= line.split('///')
                print(li)
                a=''
                w,h= 2970, 2100
                a+=str(liy.index(li[0]))+" "+str(int(li[1])/w)+" "+str(int(li[2])/h)+" "+str(int(li[3])/w)+" "+str(int(li[4])/h)+" \n"
                file2.write(a)
        