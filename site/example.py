import andrey
from andrey.data import load_dataset

df = load_dataset("sachs").data
out = andrey.pc(df)
print(out)
