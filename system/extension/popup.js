// popup.js

// Function to parse the input string
function parseString(input) {
  // Extract unique_id
  const uniqueIdStart = input.indexOf('<') + 1;
  const uniqueIdEnd = input.indexOf('>');
  const uniqueId = input.substring(uniqueIdStart, uniqueIdEnd);

  // Extract id
  const idStart = uniqueIdEnd + 1;
  const idEnd = input.indexOf(':');
  const id = input.substring(idStart, idEnd);

  // Extract cls
  const clsStart = input.indexOf('"') + 1;
  const clsEnd = input.indexOf(' -');
  const cls = parseInt(input.substring(clsStart, clsEnd));

  // Extract acsa array
  const acsaStart = input.indexOf('[') + 1;
  const acsaEnd = input.indexOf(']');
  const acsaString = input.substring(acsaStart, acsaEnd);
  const acsa = acsaString.split(',').map(Number);

  return { uniqueId, id, cls, acsa };
}

// Generate a unique ID based on the current time
function generateUniqueId() {
  return 'id-' + Date.now();
}

// document.addEventListener('DOMContentLoaded', function() {
//   document.getElementById('sendDataBtn').addEventListener('click', function() {
//     const uniqueId = generateUniqueId();
//     chrome.runtime.sendMessage({ type: "send_data" , text: uniqueId});

//     // Listen for messages from the background script
//     chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
//       if (message.type === 'result') {
//         console.log('Received WebSocket message:', message.text);
//         // Perform an action based on the message
//         result = parseString(message.text)
//         // console.log(result.uniqueId);
//         // console.log(result.id);
//         // console.log(result.cls);
//         // console.log(result.acsa);
//         if(uniqueId === result.uniqueId){
//           const resultContainer = document.getElementById('resultContainer');
//           resultContainer.innerHTML = `<p>CLS: ${result.cls}</p><p>ACSA: ${result.acsa.join(', ')}</p>`;
//           resultContainer.style.display = 'block'; // Hiển thị phần tử chứa kết quả
//         }
//       }
//     });
//   });
// });

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('sendDataBtn').addEventListener('click', function() {
    const uniqueId = generateUniqueId();
    chrome.runtime.sendMessage({ type: "send_data" , text: uniqueId});

    // Listen for messages from the background script
    chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
      if (message.type === 'result') {
        console.log('Received WebSocket message:', message.text);
        // Perform an action based on the message
        const result = parseString(message.text);
        // console.log(result.uniqueId);
        // console.log(result.id);
        // console.log(result.cls);
        // console.log(result.acsa);
        if(uniqueId === result.uniqueId){
          const resultContainer = document.getElementById('resultContainer');
          // Remove old formContainer if exists
          const oldFormContainer = document.getElementById('formContainer');
          if (oldFormContainer) {
            oldFormContainer.remove();
          }

          // Create formContainer
          const formContainer = document.createElement('div');
          formContainer.setAttribute('id', 'formContainer');

          // Create label for cls
          const clsLabel = document.createElement('label');
          clsLabel.textContent = 'CLS';
          formContainer.appendChild(clsLabel);

          // Create select for cls
          const clsSelect = document.createElement('select');
          clsSelect.innerHTML = '<option value="0">Clean</option><option value="1">Seeding</option><option value="2">Abnormal</option>';
          clsSelect.value = result.cls;
          formContainer.appendChild(clsSelect);
          formContainer.appendChild(document.createElement('br')); // Xuống dòng
          formContainer.appendChild(document.createElement('br')); // Xuống dòng
          formContainer.appendChild(document.createElement('br')); // Xuống dòng

          // Create label for acsa
          const acsaLabel = document.createElement('label');
          acsaLabel.textContent = 'ACSA';
          formContainer.appendChild(acsaLabel);
          formContainer.appendChild(document.createElement('br')); // Xuống dòng

          // Create labels and selects for acsa
          const acsaLabels = ['Title', 'Body', 'Company', 'Detail'];
          for (let i = 0; i < 4; i++) {
            const acsaLabel = document.createElement('label');
            acsaLabel.textContent = acsaLabels[i];
            formContainer.appendChild(acsaLabel);

            const acsaSelect = document.createElement('select');
            acsaSelect.innerHTML = '<option value="0">Positive</option><option value="1">Negative</option><option value="2">Not mentioned</option><option value="3">Neutral</option>';
            acsaSelect.value = result.acsa[i];
            formContainer.appendChild(acsaSelect);
            formContainer.appendChild(document.createElement('br')); // Xuống dòng
            formContainer.appendChild(document.createElement('br')); // Xuống dòng
            formContainer.appendChild(document.createElement('br')); // Xuống dòng
          }

          // Create report button
          const reportBtn = document.createElement('button');
          reportBtn.textContent = 'Report';
          reportBtn.addEventListener('click', function() {
            const clsValue = clsSelect.value;
            const acsaValues = [];
            formContainer.querySelectorAll('select').forEach(select => {
              acsaValues.push(select.value);
            });
            // const reportString = `CLS: ${clsValue}, ACSA: [${acsaValues.join(', ')}]`;
            const reportString = "{'id': " + result.id + ", 'prediction': " + `${clsValue}` + ", 'acsa': " + `[${acsaValues.join(', ')}]` + ", 'type': 'report'}";
            chrome.runtime.sendMessage({ type: "report" , text: reportString});
            console.log(reportString);
            // You can do something with the reportString here, like sending it to the server
          });
          formContainer.appendChild(reportBtn);

          resultContainer.appendChild(formContainer);
          resultContainer.style.display = 'block'; // Hiển thị phần tử chứa kết quả
        }
      }
    });
  });
});





