for z in range(4000):
    with open("C:\\Users\\vt\\Downloads\\kibo\\test\\"+str(z+36000)+".txt", "r") as file:
        i=0
        data= file.readlines()
        for line in data:
            contents=line.split(" ")
            ll=contents[0]+" "+contents[1]+" "+contents[2]+' '+str(int(contents[3])/2970)+" "+str(int(contents[4])/2100)+" \n"
            data[i]=ll
            i+=1
    print(data)
    with open("C:\\Users\\vt\\Downloads\\kibo\\test\\"+str(z+36000)+".txt", "w") as file2:
        file2.writelines(data)


