import discord
import re
import requests
import random
import time
import threading
import asyncio

client = discord.Client()

pingchannel = client.get_channel("763821567695126588")

pinginterval = 2 # seconds

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    global pinging, pinginterval, pingchannel

    if message.author == "Eon#8669":
        stopPinging()

    if message.author == client.user:
        return

    if "Eon#8669" in [str(mention) for mention in message.mentions]:
        pingean()
    
    if message.content.startswith("hey bot, ping ean!"):
        # pingchannel = message.channel
        pinging = True
        while pinging:
            await pingchannel.send('<@549394254695235594>')
            time.sleep(pinginterval)

    
    if message.content.startswith("hey bot, stop!"):
        pinging = False

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