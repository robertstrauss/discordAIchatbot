import discord
import re
import requests
import random
import time
import threading

client = discord.Client()

pinginterval = 5000 # ms

pinging = False

def pingloop():
    while pinging:
        message.channel.send('@Eon#8669')
        time.sleep(pinginterval)

pinger = threading.Thread(target = pingloop)

def startPinging():
    pinging = True
    pinger.start()


@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    global pinging, pinginterval

    if message.author == "Eon#8669":
        stopPinging()

    if message.author == client.user:
        return

    if "Eon#8669" in [str(mention) for mention in mentions]:
        pingean()
    
    if message.content.startswith("hey bot, ping ean!"):
        startPinging()
    
    if message.content.startsWith("hey bot, stop!"):
        stopPinging()

                await message.channel.send(makepenis(random.randint(0,10)))
    


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