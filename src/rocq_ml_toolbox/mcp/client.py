import asyncio
from fastmcp import Client

client = Client("http://localhost:8000/mcp?env=coq-geocoq")

async def call_tool():
    async with client:
        result = await client.call_tool("start_proof")
        print(result)
        result = await client.call_tool("run_tac", {"cmd": "split."})
        print(result)
        result = await client.call_tool("run_tac", {"cmd": "- are."})
        print(result)

asyncio.run(call_tool())