#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
post GraphQL request to load game data for a particular level
"""
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport


def query_level(level):
    """
    query the game grid data for a particular level
    :param level number of the level
    :return 2d list of grid data
    """

    transport = RequestsHTTPTransport(
        url='https://wy5oftshw4.execute-api.ap-southeast-2.amazonaws.com/dev/graphql',
        use_json=True,
        headers={
            "Content-type": "application/graphql",
        }
    )
    client = Client(transport=transport,fetch_schema_from_transport=True)
    query = gql('''
        query GetLevel($id: Int) {
            level(id: $id) {
                cells {
                    targetRow
                    steps
                    row
                    col
                }
            }
        }
    ''')
    params = {"id": level}
    result = client.execute(query, variable_values=params)
    return result['level']['cells']
