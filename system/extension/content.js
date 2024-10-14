// content.js

// Trích xuất dữ liệu phần tử và gửi qua sự kiện
function extractAndSendElements() {
  const currentUrl = String(window.location.href);
  console.log("URL của trang web hiện tại là:", currentUrl);
  if(currentUrl.includes('https://muaban.net/viec-lam/viec-tim-nguoi/')){
  const regex = /-id(\d+)/;
  const match = currentUrl.match(regex);

  if (match && match.length > 1) {
    const idNumber = match[1];
    console.log("id:", idNumber);
    chrome.runtime.sendMessage({ type: "element_data", data: idNumber });
  } else {
    console.log("Không tìm thấy id");
  }
  }
  else{
    console.log('Not recruitment website');
    chrome.runtime.sendMessage({ type: "element_data", data: 'Not recruitment website'});
  }
}

window.onload = function(){
  extractAndSendElements();
}

var currentURL = window.location.href;

// Kiểm tra sự thay đổi URL định kỳ
setInterval(function() {
  // So sánh URL hiện tại với URL mới
  if (window.location.href !== currentURL) {
      // Lưu URL mới
      currentURL = window.location.href;
      // Xử lý khi URL thay đổi
      console.log('URL đã thay đổi:', currentURL);
      extractAndSendElements();
  }
}, 100); // Kiểm tra mỗi 100 milliseconds