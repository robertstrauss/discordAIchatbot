import discord
import re
import sys, os
import requests
import random
import time
import asyncio
import pandas as pd
import modeltrain as modeltrian



client = discord.Client()

guildID = 755473404643115170

guild = None

channelID = 831188849946394654

talkchannel = None

# transcriptsfolder = 'channeltranscripts/'


@client.event
async def on_ready():
    global guild, guildID, channel, channelID
    print('We have logged in as {0.user}'.format(client))
    guild = client.get_guild(guildID)
    talkchannels = [client.get_channel(channelID)]

    # limit = int(re.findall(r'limit=(\d+)', " ".join(sys.argv))[0])

    # channelIDs = re.findall(r' (\d+) ', " ".join(sys.argv))


@client.event
async def on_message(message):
    if message.author == client.user:
        return

    if not message.channel in talkchannels:
        return
    
    response = getBotResponse(message.content)

    await message.channel.send(response)


model.loadModel()


with open('token.txt', 'r') as tokentxt:
    # asyncio.get_event_loop().create_task(pingean())
    client.run(tokentxt.read())

