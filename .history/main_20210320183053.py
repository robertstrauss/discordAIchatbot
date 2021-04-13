import discord
import re
import requests
import random
import time
import threading
import asyncio

client = discord.Client()

guildID = 755473404643115170

guild = None

pingEanID = 763821567695126588

pingchannel = None

eanID = 549394254695235594

ean = None

pinginterval = 0.1 # seconds

async def pingean():
    role = random.choice(ean.roles)
    await pingchannel.send('{}'.format(role.mention))

@client.event
async def on_ready():
    global pingchannel, ean
    print('We have logged in as {0.user}'.format(client))
    guild = client.get_guild(guildID)
    pingchannel = client.get_channel(pingEanID)
    ean = await guild.get_member(eanID)

@client.event
async def on_message(message):
    global pinging, pinginterval, pingchannel

    if message.author == "Eon#8669":
        pinging = False
        await message.channel.send('fuckin cunt')

    if message.author == client.user:
        return

    if "Eon#8669" in [str(mention) for mention in message.mentions]:
        await pingean()

    
    if message.content.startswith("hey bot, ping ean!"):
        # pingchannel = message.channel
        pinging = True
        while pinging:
            await pingean()
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