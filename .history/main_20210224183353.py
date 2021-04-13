import discord
import re
import requests
import random

client = discord.Client()



@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

def makeVertPenis(length):
    """
    vertical
    """
    thing = "( ) ( )\n"
    for i in range(length):
        thing = "  | |  \n" + thing
    thing = "  /*\  \n" + thing
    return thing

def makeHorPenis(length):
    """
    horizontal
    """
    thing = "8"
    for i in range(length):
        thing = thing + "="
    thing = thing + "D"
    return thing

makepenis = makeHorPenis

@client.event
async def on_message(message):
    global makepenis

    if message.author == client.user:
        return

    if message.content.startswith('/penismode'):
        if 'horizontal' in message.content:
            makepenis = makeHorPenis
        elif 'vertical' in message.content:
            makepenis = makeVertPenis
        
        await message.channel.send("penismode: {}".format(makepenis.__doc__))

    elif message.content.startswith('/penis'):
        person = message.author
        if len(message.mentions) > 0:
            person = message.mentions[0]

        if random.random() > 0.5:
            await message.channel.send("fuck off")
        else:
            await message.channel.send("{}'s penis is this long:".format(person))
            if str(person) == "ostrich#3524":
                await message.channel.send(makepenis(random.randint(15,20)))
            elif str(person) == "WontonFrenardo#8723":
                await message.channel.send(makepenis(1))
            elif str(person) == "Sam Wescott#8341":
                await message.channel.send(makepenis(20))
            elif str(person) == "Eon#8669":
                await message.channel.send(":speaking_head:========================00")
            else:
                await message.channel.send(makepenis(random.randint(0,10)))
    
    else:
        return
    # else:
    #     await message.channel.send("""
    #     Recognized commands:
    #     price <stock>: report the current price and change in price for a stock
    #     quote <stock>: display the current global quote for a stock
    #     track <stock> interval=<interval> change=<change>: report the quote for a stock every interval or when it changes by a certain amount
    #     """)
        

with open('apikey.txt', 'r') as apikeytxt:
    api_key = apikeytxt.read()
api_url = 'https://www.alphavantage.co/query'
def getQuote(symbol):
    data = {
        'function': 'GLOBAL_QUOTE',
        'symbol': symbol,
        'apikey': api_key
    }
    return requests.get(api_url, params=data).json()['Global Quote']



with open('token.txt', 'r') as tokentxt:
    client.run(tokentxt.read())