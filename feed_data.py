from glyphogen.dataset import get_vectorizer_data
from glyphogen.losses import find_eos


_, t = get_vectorizer_data(0,0)
count = 0
for example in iter(t):
    inputs, outputs = example
    in_sequence = inputs['target_sequence']
    out_sequence = outputs['command']
    # add a batch dimension
    in_sequence = in_sequence.unsqueeze(0)
    out_sequence = out_sequence.unsqueeze(0)
    count_in = find_eos(in_sequence)
    count_out = find_eos(out_sequence)
    assert count_out == count_in - 1
    count += count_out.item()
    print(f"Processed {count} commands", end='\r')
print("Final total: ", count)

    
