const http=require('http'), fs=require('fs'), path=require('path');
const root=path.resolve(__dirname,'..');
const types={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.woff2':'font/woff2','.md':'text/plain; charset=utf-8','.json':'application/json','.excalidraw':'application/json','.csv':'text/csv','.patch':'text/plain; charset=utf-8'};
function createServer(){return http.createServer((req,res)=>{
 let pathname;try{pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname)}catch{res.writeHead(400);return res.end()}
 if(pathname==='/')pathname='/index.html';
 const file=path.resolve(root,'.'+pathname),relative=path.relative(root,file);
 if(relative.startsWith('..')||path.isAbsolute(relative)||relative.split(path.sep).some(s=>s.startsWith('.'))){res.writeHead(403);return res.end()}
 fs.stat(file,(err,stat)=>{if(err||!stat.isFile()){res.writeHead(404);return res.end('Not found')};res.writeHead(200,{'Content-Type':types[path.extname(file)]||'application/octet-stream','X-Robots-Tag':'noindex, nofollow','Cache-Control':'no-store'});fs.createReadStream(file).pipe(res)});
})}
module.exports={createServer};
if(require.main===module)createServer().listen(8080,'127.0.0.1',()=>console.log('Private local preview: http://127.0.0.1:8080'));
