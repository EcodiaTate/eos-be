const http = require("http");
http
  .createServer((req, res) => {
    res.end("OK");
  })
  .listen(8080);
console.log(`Test server running. PID: ${process.pid}`);
