flowchart TD

    subgraph "Offline Learning (Baseline)"
        A1["CNN Train Eval"]:::offline
        A2["Models"]:::offline
        A3["Functions"]:::offline
        A4["Evaluation Task"]:::offline
        A5["Prediction Script"]:::offline
    end

    subgraph "Online Learning (System)"
        subgraph "Online Training"
            B1["Online Training 1"]:::online
            B2["Online Training 2"]:::online
            B3["Online Training 3"]:::online
            B4["Predict Module"]:::online
        end
        subgraph "Client Handling"
            C1["Client Handler"]:::client
            C2["Preprocessing"]:::client
            C3["Client Utils"]:::client
        end
        subgraph "Service Management"
            D1["Server Start"]:::service
            D2["Manager"]:::service
            D3["Shutdown"]:::service
            D4["Zoo Start"]:::service
        end
        E1["Kafka Integration"]:::kafka
    end

    subgraph "Experiments & Evaluation"
        F1["Experiments Script"]:::experiments
        F2["Result Notebook"]:::experiments
        F3["Plot Generator"]:::experiments
    end

    subgraph "Model Configurations"
        G1["Model Configs"]:::model
    end

    subgraph "Browser Extension"
        H1["Browser Extension"]:::extension
    end

    %% Data Flow Connections
    A5 -->|"feeds"| G1
    G1 -->|"supplies"| B1
    H1 -->|"initiates"| C1
    C2 -->|"preprocesses"| C1
    C3 -->|"assists"| C1
    C1 -->|"passes data to"| B4
    E1 -->|"routes streaming data to"| B4
    D1 -->|"controls"| B1
    D2 -->|"controls"| B2
    D3 -->|"controls"| B3
    D4 -->|"controls"| B4
    B4 -->|"sends output to"| F1
    F2 -->|"feedback"| B1
    F3 -->|"visualizes"| B1

    %% Class Definitions for Coloring
    classDef offline fill:#f9c,stroke:#333,stroke-width:2px;
    classDef online fill:#ccf,stroke:#333,stroke-width:2px;
    classDef experiments fill:#cfc,stroke:#333,stroke-width:2px;
    classDef model fill:#fc9,stroke:#333,stroke-width:2px;
    classDef extension fill:#9cf,stroke:#333,stroke-width:2px;
    classDef service fill:#fcf,stroke:#333,stroke-width:2px;
    classDef client fill:#cff,stroke:#333,stroke-width:2px;
    classDef kafka fill:#ff9,stroke:#333,stroke-width:2px;

    %% Click Events for Offline Learning (Baseline)
    click A1 "https://github.com/quocnguyenx43/virecax/blob/master/baseline/coms/task_1/cnn_train_eval.sh"
    click A2 "https://github.com/quocnguyenx43/virecax/blob/master/baseline/models.py"
    click A3 "https://github.com/quocnguyenx43/virecax/blob/master/baseline/functions.py"
    click A4 "https://github.com/quocnguyenx43/virecax/blob/master/baseline/run_evaluation_cls_task.py"
    click A5 "https://github.com/quocnguyenx43/virecax/blob/master/baseline/run_prediction_full.py"

    %% Click Events for Online Learning (System)
    click B1 "https://github.com/quocnguyenx43/virecax/blob/master/system/processor/online_training_1.py"
    click B2 "https://github.com/quocnguyenx43/virecax/blob/master/system/processor/online_training_2.py"
    click B3 "https://github.com/quocnguyenx43/virecax/blob/master/system/processor/online_training_3.py"
    click B4 "https://github.com/quocnguyenx43/virecax/blob/master/system/processor/predict.py"
    click C1 "https://github.com/quocnguyenx43/virecax/blob/master/system/client_handler/client_handler.py"
    click C2 "https://github.com/quocnguyenx43/virecax/blob/master/system/client_handler/preprocess.py"
    click C3 "https://github.com/quocnguyenx43/virecax/blob/master/system/client_handler/utils.py"
    click D1 "https://github.com/quocnguyenx43/virecax/blob/master/system/server_start.py"
    click D2 "https://github.com/quocnguyenx43/virecax/blob/master/system/manager.py"
    click D3 "https://github.com/quocnguyenx43/virecax/blob/master/system/shutdown.py"
    click D4 "https://github.com/quocnguyenx43/virecax/blob/master/system/zoo_start.py"
    click E1 "https://github.com/quocnguyenx43/virecax/blob/master/system/clear_data_kafka.py"

    %% Click Events for Experiments & Evaluation
    click F1 "https://github.com/quocnguyenx43/virecax/blob/master/system/experiments/get_result.py"
    click F2 "https://github.com/quocnguyenx43/virecax/blob/master/system/experiments/result.ipynb"
    click F3 "https://github.com/quocnguyenx43/virecax/blob/master/system/experiments/plots/mermaid_diagram.html"

    %% Click Events for Model Configurations
    click G1 "https://github.com/quocnguyenx43/virecax/tree/master/system/models"

    %% Click Event for Browser Extension
    click H1 "https://github.com/quocnguyenx43/virecax/tree/master/system/extension"


