from __future__ import print_function

import os.path
from pprint import pprint
import numpy as np

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

class Sheet:
    def __init__(self, spreadsheet_id):
        self.sheet = None
        self.spreadsheet_id = spreadsheet_id

    def connect_sheet(self):
        creds = None
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.json'):
            creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.json', 'w') as token:
                token.write(creds.to_json())

        try:
            service = build('sheets', 'v4', credentials=creds)

            # Call the Sheets API
            self.sheet = service.spreadsheets()
        except HttpError as err:
            print(err)

    def read_rows_sheet(self, sheet_range):
        request = self.sheet.values().get(
            spreadsheetId = self.spreadsheet_id,
            range = sheet_range
        )
        response = request.execute()

        rows = response.get('values', [])
        return rows

    def write_sheet(self, sheet_range, value_input_option, value_range_body):
        try:
            request = self.sheet.values().update(
                spreadsheetId = self.spreadsheet_id,
                range = sheet_range,
                valueInputOption = value_input_option,
                body = value_range_body
            )
            response = request.execute()
        except HttpError as err:
            print(err)

    def update_sheet(self, sheet_name, df, index=True, value_input_option='USER_ENTERED'):
        if index:
            values = [list(df.index.names) + df.columns.tolist()]
            if len(df.index.names) > 1:
                values += [
                    list(_index) + value for _index, value in zip(
                        df.index.tolist(), df.replace(
                            [np.inf, -np.inf], np.nan
                        ).fillna('').values.tolist()
                    )
                ]
            else:
                values += [
                    [_index] + value for _index, value in zip(
                        df.index.tolist(), df.replace(
                            [np.inf, -np.inf], np.nan
                        ).fillna('').values.tolist()
                    )
                ]
        else:
            values = [df.columns.tolist()]
            values += [
                value for value in df.replace(
                    [np.inf, -np.inf], np.nan
                ).fillna('').values.tolist()
            ]

        num_cols = len(values[0])
        sheet_range = f'{sheet_name}!R1C1:C{num_cols}'
        rows = self.read_rows_sheet(sheet_range)
        self.write_sheet(
            sheet_range,
            value_input_option,
            {'values': [['' for _ in range(num_cols)] for _ in range(len(rows))]}
        )
        self.write_sheet(
            sheet_range,
            value_input_option,
            {'values': values}
        )


def main():
    pass

if __name__ == '__main__':
    main()