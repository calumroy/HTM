#!/usr/bin/python

"""
HTM card pattern input creator and command input.

This code creates a card pattern that can be fed into the HTM network.

author: Calum Meiklejohn
website: calumroy.com
last edited: August 2013
"""
import numpy as np
import math
import random


##def createNewHand(cardSize):
##    # Creates an input grid with two cards which are active.
##    # There are three suits so range is 0,1,2,3 and 13 cards 0 to 12.
##    cardNum1,cardSuit1 = round(random.uniform(0,12)),round(random.uniform(0,3))
##    cardNum2,cardSuit2 = round(random.uniform(0,12)),round(random.uniform(0,3))
##    # Make sure the same cards aren't chosen.
##    while (cardNum1,cardSuit1)==(cardNum2,cardSuit2):
##        cardNum2,cardSuit2 = round(random.uniform(0,12)),round(random.uniform(0,3))
##    inputGrid=np.array([[0 for i in range(cardSize*4)] for j in range(cardSize*13)])
##    for x in range(len(inputGrid[0])):
##        for y in range(len(inputGrid)):
##            if x>=cardSuit1*cardSize and x<(cardSuit1+1)*cardSize and y>=cardNum1*cardSize and y<(cardNum1+1)*cardSize:
##                #print (y,x)
##                inputGrid[y][x]=1
##            if x>=cardSuit2*cardSize and x<(cardSuit2+1)*cardSize and y>=cardNum2*cardSize and y<(cardNum2+1)*cardSize:
##                inputGrid[y][x]=1
##    return inputGrid

def newOppCard(numCards):
    return round(random.uniform(0,numCards))
def oppPlay(oppCard,width,height):
    #commandHeight = round(height/3)
    commandHeight = 4
    # Create a type of personality for the opponent
    aggresiveness = 0
    chance = round(random.uniform(0,10))
    play='fold'
    if oppCard >= 4:
        play='raise' 
    elif chance > (10-aggresiveness):
        play='raise'
    else:
        if oppCard>1:
            play='check'
    # Create the input grid for the opponents play. 
    inputGrid=np.array([[0 for i in range(width)] for j in range(height)])
    for x in range(len(inputGrid[0])):
        for y in range(len(inputGrid)):
            if play=='raise':
                if y<commandHeight:
                    inputGrid[y][x]=1
            if play=='check':
                if y>=commandHeight and y<(2*commandHeight):
                    inputGrid[y][x]=1
            if play=='fold':
                if y>=(2*commandHeight) and y<(3*commandHeight):
                    inputGrid[y][x]=1    
    return inputGrid,play


def createNewSimpleHand(cardSize,numCards):
    # Creates an input grid with two cards which are active.
    cardNum1 = round(random.uniform(0,numCards-1))
    inputGrid=np.array([[0 for i in range(cardSize)] for j in range(cardSize*numCards)])
    for x in range(len(inputGrid[0])):
        for y in range(len(inputGrid)):
            if y>=cardNum1*cardSize and y<(cardNum1+1)*cardSize:
                #print (y,x)
                inputGrid[y][x]=1
    return inputGrid,cardNum1
def newCommInput(width,numCommRows,command):
    newComm = np.array([[0 for i in range(width)] for j in range(numCommRows)])
    for x in range(len(newComm[0])):
        for y in range(len(newComm)):
            if command=='check':
                if x>=round(width/3) and x<round(2*width/3):
                    #print (y,x)
                    newComm[y][x]=1 
            if command=='raise':
                if x<round(width/3):
                    #print (y,x)
                    newComm[y][x]=1
            if command=='fold':
                if x>=round(2*width/3):
                    #print (y,x)
                    newComm[y][x]=1
            if command=='none':
                pass
    return newComm
def randomCommand():
    ran_command = round(random.uniform(0,2)) 
    if ran_command ==2:
        command = 'raise'
        print "     chose raise"
    if ran_command ==1:
        command = 'check'
        print "     chose check"
    if ran_command ==0:
        command = 'fold'
        print "     chose fold"
    return command

def updateMoney(width,height,money,command,card,oppCard,oppPlay):
    # Creates an input grid depending on the money, the previous bet and the opponents bet.
    # Rules for simple poker:
    #
    #   HTM (you) commands|Opponents Plays  | money won if card is higher than oppCard
    #               raise | fold            | 1 won (always)
    #               raise | check           | 2 won or 2 lost 
    #               raise | raise           | 3 won or 3 lost
    #               check | fold            | 1 win (always)
    #               check | check           | 1 win or 1 lost
    #               check | raise           | 2 win or 2 lost
    #               fold | fold            | 0 win (always)
    #               fold | check           | 0 lost (always)
    #               fold | raise           | 0 win or 0 lost
    # If you lose you lose 1/height percent (1 row) of your money. If you win you
    # gain 1/height percent.
    if oppPlay=='fold' and command!='fold':
        money += 1
    if command=='raise' and oppPlay=='check':
        if card>oppCard:
            money += 2
        else:
            money -= 2
    if command=='raise' and oppPlay=='raise':
        if card>oppCard:
            money += 3
        else:
            money -= 3
    if command=='check' and oppPlay=='check':
        if card>oppCard:
            money += 1
        else:
            money -= 1
    if command=='check' and oppPlay=='raise':
        if card>oppCard:
            money += 2
        else:
            money -= 2

    if money<0:
        money = height
    if money>height:
        money = 0
    inputGrid=np.array([[0 for i in range(width)] for j in range(height)])
    for x in range(len(inputGrid[0])):
        for y in range(len(inputGrid)):
            # Draw the money from the bottom up.
            if y==(height-money):
                #print (y,x)
                inputGrid[y][x]=1
    return inputGrid,money