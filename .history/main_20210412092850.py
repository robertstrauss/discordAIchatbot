import discord
import re
import requests
import random
import time
import asyncio
import pytorch

client = discord.Client()

guildID = 755473404643115170

guild = None

channelID = 824321516891537429

talkchannel = None

pinginterval = 1 # seconds

pinging = False

async def pingean():
    while True:

        await asyncio.sleep(pinginterval)
        print(pinging)
        if (pinging):
            role = random.choice(ean.roles)
            await pingchannel.send('{}'.format(role.mention))


def startpingean():
    global pinging
    pinging = True

def stoppingean():
    global pinging
    pinging = False


@client.event
async def on_ready():
    global guild, pingchannel, ean, logan, sam
    print('We have logged in as {0.user}'.format(client))
    guild = client.get_guild(guildID)
    pingchannel = client.get_channel(pingEanID)
    ean = await guild.fetch_member(eanID)
    logan = await guild.fetch_member(loganID)
    sam = await guild.fetch_member(samID)

@client.event
async def on_message(message):
    global pinging, pinginterval, pingchannel, ean

    if pinging and message.author == ean:
        stoppingean()
        await message.channel.send('fuckin cunt')

    if message.author == client.user:
        return

    if "Eon#8669" in [str(mention) for mention in message.mentions]:
        startpingean()

    
    if message.content.startswith("hey bot, ping ean!"):
        startpingean()


    if message.content.startswith("hey bot, ping logan!"):
        ean = logan
        startpingean()

    if message.content.startswith("hey bot, ping sam!"):
        ean = sam
        startpingean()
    
    if message.content.startswith("hey bot, stop!"):
        stoppingean()
    


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
    asyncio.get_event_loop().create_task(pingean())
    client.run(tokentxt.read())
