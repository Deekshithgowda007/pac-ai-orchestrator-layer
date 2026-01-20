from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi import Request, Response
import time

REQS = Counter("orchestrator_requests_total","Total requests",["path","method","code"])
LAT  = Histogram("orchestrator_request_seconds","Latency",["path"])

async def metrics_app(environ, start_response):
    data = generate_latest()
    start_response("200 OK", [("Content-Type", CONTENT_TYPE_LATEST)])
    return [data]

async def metrics_endpoint(request: Request):
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

async def metrics_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    LAT.labels(request.url.path).observe(time.time()-start)
    REQS.labels(request.url.path, request.method, str(response.status_code)).inc()
    return response
