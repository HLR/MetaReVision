"""
I should clean the VLModels:
1. VLBERT
2. LXMERT
3. VisualBert (we should have that later)

all the models should have the same API


------------------------------------------------------


External Bert Knowledge:

Case 1:
#  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
#  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1

Case 2:
#  tokens:   [CLS] the dog is hairy . [SEP]
#  type_ids: 0     0   0   0  0     0 0

"type_ids" are used to indicate whether this is the first

-------------
ID
Segment
Mask
-------------
"""

