import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
import redis

# Connecting to Redis server on localhost
redis_client = redis.Redis(host='localhost', port=6379, db=0)
print('Connection Done!')
from redisvl.extensions.cache.llm import SemanticCache
from redisvl.utils.vectorize import HFTextVectorizer

llmcache = SemanticCache(
    name="llmcache",                                          # underlying search index name
    redis_url="redis://localhost:6379",                       # redis connection url string
    distance_threshold=0.1,                                   # semantic cache distance threshold
    vectorizer=HFTextVectorizer("redis/langcache-embed-v1"),  # embdding model
)

question = "What is the capital of France?"
if response := llmcache.check(prompt=question):
    print(response)
else:
    print("Empty cache")

llmcache.store(
    prompt=question,
    response="Paris",
)
if response := llmcache.check(prompt=question, return_fields=["prompt", "response", "metadata"]):
    print(response)
else:
    print("Empty cache")

print("differnt question")
# Check for a semantically similar result
question = "What actually is the capital of France?"
if response := llmcache.check(prompt=question)[0]['response']:
    print(response)
