id	depot_code	depot	fleet_no	route_service_type	trip_no	ticket_no	date	time	ticket_type	source	destination	passenger_count	luggage_count	passenger_amount	luggage_amount	driver_no	conductor_no	waybill_no	schedule_no	payment_type	direction	Trip_Origin	Trip_Destination	total_ticket_amount	KM	duty_date	from_stage	to_stage	add_trip	device_serial_number	NCMC_Recharge_Amount	pine_labs_transaction_no	merchant_id	rrn_number	terminal_id
491788	AY	Ayanavaram Depot	K0297	Ordinary	1	1	01-03-2026	03:29:49	Adult Ticket	AYANAVARAM B.T	WHEELS INDIA RD.JN	1	0	7	0	D51192	C53210	T1772304238392452658	O-48C-W-AM	CASH	48CCT - UP	AYANAVARAM B.T	M.G.R.KOYAMBEDU	7	9.7	01-03-2026	4	7	N	1493453157					
491897	AY	Ayanavaram Depot	K0297	Ordinary	1	2	01-03-2026	03:30:24	Adult Ticket	AYANAVARAM B.T	M.G.R.KOYAMBEDU	1	0	10	0	D51192	C53210	T1772304238392452658	O-48C-W-AM	CASH	48CCT - UP	AYANAVARAM B.T	M.G.R.KOYAMBEDU	10	9.7	01-03-2026	4	10	N	1493453157					
492002	AY	Ayanavaram Depot	K0297	Ordinary	1	3	01-03-2026	03:30:47	Adult Ticket	AYANAVARAM B.T	M.G.R.KOYAMBEDU	1	0	10	0	D51192	C53210	T1772304238392452658	O-48C-W-AM	CASH	48CCT - UP	AYANAVARAM B.T	M.G.R.KOYAMBEDU	10	9.7	01-03-2026	4	10	N	1493453157					


# Generated Project Data Schema


## `Data/ETM/Ticket-Data-2026-03-01.csv`

### Columns & Data Types

| Column | Data Type |
| :--- | :--- |
| `id` | `int64` |
| `depot_code` | `str` |
| `depot` | `str` |
| `fleet_no` | `str` |
| `route_service_type` | `str` |
| `trip_no` | `int64` |
| `ticket_no` | `int64` |
| `date` | `str` |
| `time` | `str` |
| `ticket_type` | `str` |
| `source` | `str` |
| `destination` | `str` |
| `passenger_count` | `int64` |
| `luggage_count` | `int64` |
| `passenger_amount` | `float64` |
| `luggage_amount` | `float64` |
| `driver_no` | `str` |
| `conductor_no` | `str` |
| `waybill_no` | `str` |
| `schedule_no` | `str` |
| `payment_type` | `str` |
| `direction` | `str` |
| `Trip_Origin` | `str` |
| `Trip_Destination` | `str` |
| `total_ticket_amount` | `float64` |
| `KM` | `float64` |
| `duty_date` | `str` |
| `from_stage` | `int64` |
| `to_stage` | `float64` |
| `add_trip` | `str` |
| `device_serial_number` | `int64` |
| `NCMC_Recharge_Amount` | `float64` |
| `pine_labs_transaction_no` | `float64` |
| `merchant_id` | `float64` |
| `rrn_number` | `float64` |
| `terminal_id` | `float64` |

### Sample Rows (First 2 Rows)

| id | depot_code | depot | fleet_no | route_service_type | trip_no | ticket_no | date | time | ticket_type | source | destination | passenger_count | luggage_count | passenger_amount | luggage_amount | driver_no | conductor_no | waybill_no | schedule_no | payment_type | direction | Trip_Origin | Trip_Destination | total_ticket_amount | KM | duty_date | from_stage | to_stage | add_trip | device_serial_number | NCMC_Recharge_Amount | pine_labs_transaction_no | merchant_id | rrn_number | terminal_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 491788 | AY | Ayanavaram Depot | K0297 | Ordinary | 1 | 1 | 01-03-2026 | 03:29:49 | Adult Ticket | AYANAVARAM B.T | WHEELS INDIA RD.JN | 1 | 0 | 7.0 | 0.0 | D51192 | C53210 | T1772304238392452658 | O-48C-W-AM | CASH | 48CCT - UP | AYANAVARAM B.T | M.G.R.KOYAMBEDU | 7.0 | 9.7 | 2026-03-01 | 4 | 7.0 | N | 1493453157 | nan | nan | nan | nan | nan |
| 491897 | AY | Ayanavaram Depot | K0297 | Ordinary | 1 | 2 | 01-03-2026 | 03:30:24 | Adult Ticket | AYANAVARAM B.T | M.G.R.KOYAMBEDU | 1 | 0 | 10.0 | 0.0 | D51192 | C53210 | T1772304238392452658 | O-48C-W-AM | CASH | 48CCT - UP | AYANAVARAM B.T | M.G.R.KOYAMBEDU | 10.0 | 9.7 | 2026-03-01 | 4 | 10.0 | N | 1493453157 | nan | nan | nan | nan | nan |



## `Data/Public Transport/Metro Rail/Operational_Metro_Stns.shp`

*(Requested reference: `Public Transport/Metro Rail/Operational_Metro_Stns.shp`)*

### Columns & Data Types

| Column | Data Type |
| :--- | :--- |
| `id` | `object` |
| `Name` | `str` |
| `descriptio` | `object` |
| `timestamp` | `object` |
| `begin` | `object` |
| `end` | `object` |
| `altitudeMo` | `object` |
| `tessellate` | `int64` |
| `extrude` | `int64` |
| `visibility` | `int64` |
| `drawOrder` | `float64` |
| `icon` | `object` |
| `geometry` | `geometry` |

### Sample Rows (First 2 Rows)

| id | Name | descriptio | timestamp | begin | end | altitudeMo | tessellate | extrude | visibility | drawOrder | icon | geometry |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| None | Phase 1 C2-Stations 1 | None | None | None | None | None | -1 | 0 | -1 | nan | None | POINT Z (80.212063 13.050681 0) |
| None | Phase 1 C2-Stations 2 | None | None | None | None | None | -1 | 0 | -1 | nan | None | POINT Z (80.232655 13.075246 0) |



## `Data/Public Transport/MTC/routes_merged.shp`

*(Requested reference: `Public Transport/MTC/routes_merged.shp`)*

### Columns & Data Types

| Column | Data Type |
| :--- | :--- |
| `route_num` | `str` |
| `source` | `str` |
| `destinat` | `str` |
| `route_id` | `float64` |
| `poly_idx` | `int64` |
| `Corridor` | `object` |
| `geometry` | `geometry` |

### Sample Rows (First 2 Rows)

| route_num | source | destinat | route_id | poly_idx | Corridor | geometry |
| --- | --- | --- | --- | --- | --- | --- |
| 1C | ENNORE | THIRUVANMIYUR | 2569.0 | 1 | None | LINESTRING (80.32074 13.21469, 80.32047 13.21482, 80.3203 13.21457, 80.31996 13.... |
| 1D | ENNORE | THIRUVANMIYUR | 2571.0 | 1 | None | LINESTRING (80.32047 13.21482, 80.32053 13.21504, 80.32058 13.21551, 80.32057 13... |

