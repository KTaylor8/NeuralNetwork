import random
import math
import time

print("This is a short mini game")
continuegame = "y"
time.sleep(1)

while continuegame == "y":
    print("you are going to fight a goblin")
    time.sleep(1)
    yourhealth = int(20)
    orchealth = int(20)
    turn = 0

    while yourhealth > 0 and orchealth > int(0):
        turn = turn + 1

        if turn % 2 != 0:
            print("It's your turn!")
            weaponchoice = input("Choose weapon: sword or magic")

            if weaponchoice == ("sword"):
                damage = random.randint(0, 5)
                print("You hit the orc with your sword, and dealt ", str(damage), "damage to it!")
                orchealth = orchealth - damage
                time.sleep(1)
            
            elif weaponchoice == ("magic"):
                damage = random.randint(0, 5)
                print("You threw a fireball at the orc, and dealth ", str(damage),"damage to it!")
                orchealth = orchealth - damage
                time.sleep(1)


        elif turn % 2 == 0:
            print("It's the orc's turn!")
            orcchoice = random.randint(0, 1)
            time.sleep(1)

            if orcchoice == 0:
                yourdamage = random.randint(0, 5)
                print("He swung his morningstar at you! You took", str(yourdamage), "damage")
                yourhealth = yourhealth - yourdamage
                time.sleep(1)

            elif orcchoice == 1:
                yourdamage = random.randint(0, 5)
                print("The orc hit you with his fists! You took ", str(yourdamage), "damage")
                yourhealth = yourhealth - yourdamage
                time.sleep(1)

    continuegame = input("Do you want to continue? y/n")
    time.sleep(2)
    if continuegame == "y":
        continue
    elif continuegame == "n":
        print("game over!")
        break