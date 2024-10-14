# mermaid_code = """
# graph TD;
#     A[Input] --> B["Pretrained Model"]
#     B -- or --> C2["FC1<br>(hidden_size=Pretrained's<br>hidden_size,<br>1024)"]
#     C2 -- ReLU<br>Dropout --> D2["FCs 2<br>(1024, 512)"]
#     B -- or --> C["LSTM1<br>(hidden_size=Pretrained's<br>hidden_size,<br>num_layers=1)"]
#     C --> D["LSTM2<br>(hidden_size=128,<br>num_layers=1)"]
#     B -- or --> C1["CNN1<br>(in_channels=1,<br>out_channels=768,<br>kernel_size=256,<br>padding=32)"]
#     C1 -- ReLU --> D1["CNN2<br>(in_channels=768,<br>out_channels=384,<br>kernel_size=128,<br>padding=16)"]
#     D --> E["FCs 3<br>(512)"]
#     D2 -- Dropout --> E["FCs 3<br>(512)"]
#     D1 -- Max Pooling,<br>ReLU --> E["FCs 3<br>(512)"]
#     E -- Dropout --> G["FCs 4<br>(256)"]
#     G -- Dropout --> I["FCs 5<br>(3)"]
#     I -- Softmax --> K[Output]
# """

mermaid_code ="""
graph TD;
    A0["Input"] --> A["Pretrained Model"] 
    A -- or --> B2["FCs ModuleList 1(4)<br>hidden_size=Pretrained's<br>hidden_size,<br>512"]
    A -- or --> B["LSTM ModuleList 1(4)<br>hidden_size=Pretrained's<br>hidden_size,<br>num_layers=1"]
    B --> C["LSTM ModuleList 2(4)<br>hidden_size=128,<br>num_layers=1"]
    A -- or --> B1["CNN ModuleList 1(4)<br>(in_channels=1,<br>out_channels=768,<br>kernel_size=256,<br>padding=32)"]
    B1 -- ReLU --> C1["CNN ModuleList 2(4)<br>(in_channels=768,<br>out_channels=384,<br>kernel_size=128,<br>padding=16)"]
    C -- Dropout, <br> ReLU --> D["ModuleList FCs 2 <br> (batch_size, 4, 512)"]
    C1 -- Max Pooling,<br>ReLU --> D
    B2 -- ReLU<br>Dropout --> D
    D -- Dropout, <br> ReLU --> E["ModuleList FCs 3 <br> (batch_size, 4, 4)"]
    E -- Softmax,<br>Stack,<br>Transpose --> F["Output <br>(batch_size, 4, 4)"]
"""


html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <script type="module">
        import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.esm.min.mjs';
        mermaid.initialize({{ startOnLoad: true }});
    </script>
</head>
<body>
    <div class="mermaid">
    {mermaid_code}
    </div>
</body>
</html>
"""

# Write the HTML content to a file
with open(r'E:\Learning\Docker_basic\basic_kafka\kltn\experiments\plots\mermaid_diagram.html', 'w') as f:
    f.write(html_content)

# Provide instructions for opening the HTML file
print("HTML file created. Open 'mermaid_diagram.html' in a web browser to view the diagram.")