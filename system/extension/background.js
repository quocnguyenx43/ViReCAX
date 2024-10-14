// background.js
var elementS;

// Function to send a message to the popup
function sendMessageToPopup(message) {
  chrome.runtime.sendMessage(message);
}



chrome.runtime.onMessage.addListener(function(message, sender, sendResponse) {
  var ws = new WebSocket('ws://localhost:8765');
  
  ws.onmessage = function(event){
    sendMessageToPopup({ type: 'result', text: event.data });
  };
  
  if (message.type === "element_data") {
    elementS = message.data;
    }
  else if (message.type === "send_data") {
    if(elementS != 'Not recruitment website'){

      ws.onopen = function(event) {
        console.log("Connected to WebSocket server");
        mess = "{'correlation_id': '" + message.text + "' , 'id': '" + elementS + "', 'type': 'request'}"
        ws.send(mess);
        console.log("Sent");
      }
    }
  }
  else if (message.type === "report"){
    ws.onopen = function(event) {
      ws.send(message.text)
    }
  }
});
