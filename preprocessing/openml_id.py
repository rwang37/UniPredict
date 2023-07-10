"""
This module sends request to openml in order to fetch the information of all
datasets. After that, we filter dataset by their quality and the number of features.
All the qualified dataset are included in the id dict. This dict will be used 
by other modules to fetch further information (annotations, data, etc.)

function(s):
 - fetch_id_dict: fetch the information from openml, and convert the json
   response into a dict of ('dataset_name'/'dataset_id').
"""
import json
import requests


def fetch_id_dict():

    url = "https://www.openml.org/es/data/data/_search?type=data&sort=runs"

    payload = json.dumps({
    "from": 0,
    "size": 10000,
    "query": {
        "bool": {
        "must": {
            "match_all": {}
        },
        "filter": [
            {
            "term": {
                "status": "active"
            }
            }
        ],
        "should": [
            {
            "term": {
                "visibility": "public"
            }
            }
        ],
        "minimum_should_match": 1
        }
    },
    "aggs": {
        "type": {
        "terms": {
            "field": "_type"
        }
        }
    },
    "_source": [
        "data_id",
        "name",
        "version",
        "format",
        "description",
        "qualities.NumberOfInstances",
        "qualities.NumberOfFeatures",
        "qualities.NumberOfClasses",
        "qualities.NumberOfMissingValues",
        "runs",
        "nr_of_likes",
        "nr_of_downloads",
        "reach",
        "impact",
        "status",
        "date",
        "url"
    ],
    "sort": {
        "runs": {
        "order": "desc"
        }
    }
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("GET", url, headers=headers, data=payload)

    json_data = response.json()
    openml_ids = {}

    # OpenML includes duplicated datasets. If two datasets have the same name,
    # we choose the one with the most visits. Since the fetched data are alreadly
    # sorted by this, we only need to reverse the list so that most frequent
    # dataset's id will cover others' (if there are duplications) when converting
    # it to a dict item, as it is placed in the last.
    json_data = json_data['hits']['hits']
    json_data.reverse()

    for item in json_data:
        if 'qualities' not in item['_source'].keys()\
            or item['_source']['qualities']['NumberOfFeatures'] > 25:
            continue
        id = item['_source']['data_id']
        name = item['_source']['name']
        openml_ids[name] = id

    assert openml_ids['credit-g'] == '31', 'Problems taken place in fetching the id dict.'
    return openml_id

if __name__ == '__main__':
    # example usage
    ID_DICT = fetch_id_dict()
