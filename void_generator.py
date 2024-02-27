import numpy as np
import glob
import cv2

import pyarrow as pa
import pyarrow.parquet as pq

void50_raw = pq.read_table("void/void-50.parquet")
void_filename = void50_raw.column_names
void50 = np.stack(
    [np.array(void50_raw[file]).reshape(50, 50) for file in void_filename]
)

print(np.shape(void50))

void20 = []
void15 = []
void10 = []
void07 = []

for i, v50 in enumerate(void50):
    v20 = cv2.resize(v50, (20, 20))
    _, v20 = cv2.threshold(v20, 127, 255, cv2.THRESH_BINARY)
    void20.append(v20)


    v15 = cv2.resize(v50, (15, 15))
    _, v15 = cv2.threshold(v15, 127, 255, cv2.THRESH_BINARY)
    void15.append(v15)


    v10 = cv2.resize(v50, (10, 10))
    _, v10 = cv2.threshold(v10, 127, 255, cv2.THRESH_BINARY)
    void10.append(v10)


    v07 = cv2.resize(v50, (7, 7))
    _, v07 = cv2.threshold(v07, 127, 255, cv2.THRESH_BINARY)
    void07.append(v07)

void20 = np.stack(void20)
void15 = np.stack(void15)
void10 = np.stack(void10)
void07 = np.stack(void07)

dvoid20 = {f"void-20-{k[8:12]}.png": v.reshape(-1) for k, v in zip(void_filename, void20)}
dvoid15 = {f"void-15-{k[8:12]}.png": v.reshape(-1) for k, v in zip(void_filename, void15)}
dvoid10 = {f"void-10-{k[8:12]}.png": v.reshape(-1) for k, v in zip(void_filename, void10)}
dvoid07 = {f"void-07-{k[8:12]}.png": v.reshape(-1) for k, v in zip(void_filename, void07)}

tvoid20 = pa.table(dvoid20)
tvoid15 = pa.table(dvoid15)
tvoid10 = pa.table(dvoid10)
tvoid07 = pa.table(dvoid07)

pq.write_table(tvoid20, "void-20.parquet")
pq.write_table(tvoid15, "void-15.parquet")
pq.write_table(tvoid10, "void-10.parquet")
pq.write_table(tvoid07, "void-07.parquet")