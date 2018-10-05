import math

restart = "y" #auto set so loop begins

while restart == "y": #creates loop to restart

    print("This is a standard deviation calculator")

    list = [(float(x)) for x in input().split()] #input list from user
    print(list)

    mean = float((sum(list))/(len(list))) #finds location of mean

    listb = [(float(x - mean)) for x in list] #subtracts mean from each number in list
    print(listb)

    listc = [(float((x)**2)) for x in listb] #squares each subtraction in listb
    print(listc)

    average = ((sum(listc))) / len(listc) #takes average of list c
    print("The variance is ", average)

    deviation = (math.sqrt(average)) #square root of average
    print("The deviation is ", deviation)

    restart = input("Restart calculator y/n ")
    if restart == "y": #if yes, continue loop
        continue
    elif restart == "n": #if no, break loop
        break