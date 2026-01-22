import asyncio
import time

async def work(i: int) -> int:
    await asyncio.sleep(i)
    return i

async def work2(i: int) -> int:
    await asyncio.sleep(i / 2)
    return i / 2

async def main():
    start = time.time()
    results = await asyncio.gather(*[work(i) for i in range(10)])
    print("Time taken for work: ", time.time() - start)
    print(results)
    results2 = await asyncio.gather(*[work2(i) for i in range(10)])
    print("Time taken for work2: ", time.time() - start)
    print(results2)

if __name__ == "__main__":
    asyncio.run(main())