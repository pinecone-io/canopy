import asyncio
import functools
from itertools import islice
from typing import Any, Callable, Optional, Tuple, Union

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.local.async_qdrant_local import AsyncQdrantLocal
import logging

logger = logging.getLogger(__name__)


def sync_fallback(method: Callable) -> Callable:
    @functools.wraps(method)
    async def wrapper(self, *args, **kwargs):
        if self._async_client is None or isinstance(
            self._async_client._client, AsyncQdrantLocal
        ):
            sync_method_name = method.__name__[1:]

            logger.warning(
                f"{method.__name__}() cannot be used for QdrantLocal. "
                f"Falling back to {sync_method_name}()"
            )
            loop = asyncio.get_event_loop()

            call = functools.partial(getattr(self, sync_method_name), *args, **kwargs)
            return await loop.run_in_executor(None, call)
        else:
            return await method(self, *args, **kwargs)

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
    force_disable_check_same_thread: bool = False,
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
        force_disable_check_same_thread=force_disable_check_same_thread,
        **kwargs,
    )

    if location == ":memory:" or path is not None:
        # In-memory Qdrant doesn't interoperate with Sync and Async clients
        # We fallback to sync operations in this case using @utils.sync_fallback
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
            force_disable_check_same_thread=force_disable_check_same_thread,
            **kwargs,
        )

    return sync_client, async_client


def batched(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    Based on itertools.batched() from Python 3.12
    """
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch
