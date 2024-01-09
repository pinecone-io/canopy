import asyncio
import functools
from itertools import islice
from typing import Any, Callable, Optional, Tuple, Union

from qdrant_client import AsyncQdrantClient, QdrantClient


def sync_fallback(method: Callable) -> Callable:
    @functools.wraps(method)
    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await method(self, *args, **kwargs)
        except NotImplementedError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, getattr(self, method.__name__[1:]), *args, **kwargs
            )

    return wrapper

def generate_clients(
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: Optional[int] = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: Optional[bool] = None,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[QdrantClient, Union[AsyncQdrantClient, None]]:
        sync_client = QdrantClient(
            location=location,
            url=url,
            port=port,
            grpc_port=grpc_port,
            prefer_grpc=prefer_grpc,
            https=https,
            api_key=api_key,
            prefix=prefix,
            timeout=timeout,
            host=host,
            path=path,
            **kwargs,
        )

        if location == ":memory:" or path is not None:
            # Local Qdrant cannot co-exist with Sync and Async clients
            # We fallback to sync operations in this case
            async_client = None
        else:
            async_client = AsyncQdrantClient(
                location=location,
                url=url,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                https=https,
                api_key=api_key,
                prefix=prefix,
                timeout=timeout,
                host=host,
                path=path,
                **kwargs,
            )

        return sync_client, async_client

def batched(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.

    """
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
