"""Quick smoke-test for embed-v-4-0 via Foundry project endpoint."""

import asyncio

from ai_search.clients import get_foundry_embed_client


async def test():
    client = get_foundry_embed_client()

    # Test text embedding
    print("=== Text embedding ===")
    resp = await client.embeddings.create(
        model="embed-v-4-0",
        input=["A mountain landscape at sunset"],
        dimensions=1024,
    )
    vec = resp.data[0].embedding[:5]
    print(f"OK — dim={len(resp.data[0].embedding)}, preview={vec}")

    # Test image embedding via data URI (download + base64)
    print("\n=== Image embedding (data URI) ===")
    import base64
    import httpx

    dl = httpx.Client(follow_redirects=True)
    img = dl.get("https://images.unsplash.com/photo-1519681393784-d120267933ba?w=512")
    b64 = base64.b64encode(img.content).decode()
    data_uri = f"data:{img.headers.get('content-type', 'image/jpeg')};base64,{b64}"
    print(f"Downloaded image: {len(img.content)} bytes")

    resp2 = await client.embeddings.create(
        model="embed-v-4-0",
        input=[data_uri],
        dimensions=1024,
    )
    vec2 = resp2.data[0].embedding[:5]
    print(f"OK — dim={len(resp2.data[0].embedding)}, preview={vec2}")


if __name__ == "__main__":
    asyncio.run(test())
