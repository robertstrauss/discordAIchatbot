import discord
import re
import requests
import random
import time
import threading
import asyncio

client = discord.Client()

pingchannel = client.get_channel("763821567695126588")

pinginterval = 5 # ms

pinging = False

async def _pingloop():
    while pinging:
        print('============ BRUH')
        await pingchannel.send('@Eon#8669')
        time.sleep(pinginterval)

def pingloop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while pinging:
        print("========== BRUH")
        loop.run_until_complete(pingchannel.send('@Eon#8669'))
        time.sleep(pinginterval)
    
    loop.close()


pinger = threading.Thread(target = pingloop)

def startPinging():
    global pinger
    pinging = True
    pinger = threading.Thread(target = asyncio.run, args = (pingloop(),))
    pinger.start()

def stopPinging():
    pinging = False
    pinger.join()


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

    if "Eon#8669" in [str(mention) for mention in message.mentions]:
        pingean()
    
    if message.content.startswith("hey bot, ping ean!"):
        startPinging()
    
    if message.content.startswith("hey bot, stop!"):
        stopPinging()

                # await message.channel.send(makepenis(random.randint(0,10)))
    


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