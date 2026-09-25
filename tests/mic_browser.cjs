// Requires Playwright; feeds a WAV through Chrome's simulated microphone.
const {chromium}=require('playwright');
const assert=require('node:assert/strict');
const path=require('node:path');
const url=process.env.SHENAVA_URL || 'http://127.0.0.1:3000';
if (!process.env.MIC_TEST_WAV) throw new Error('Set MIC_TEST_WAV to a Persian WAV fixture');
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH,headless:true,args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',`--use-file-for-fake-audio-capture=${path.resolve(process.env.MIC_TEST_WAV)}`]});
 try {
 const context=await browser.newContext();
 await context.addInitScript(()=>{const original=navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);navigator.mediaDevices.getUserMedia=async(...args)=>{window.testStream=await original(...args);return window.testStream;};});
 const page=await context.newPage();const errors=[],updates=[];
 page.on('pageerror',e=>errors.push(e.message));
 page.on('websocket',ws=>ws.on('framereceived',({payload})=>{try{const m=JSON.parse(payload);if(m.text)updates.push(m);}catch{}}));
 await page.goto(url);
 await page.getByRole('button',{name:'Start microphone'}).click();
 await page.waitForFunction(()=>document.getElementById('transcript').value.length>30,{},{timeout:20000});
 assert(await page.getByRole('button',{name:'Stop',exact:true}).isEnabled());
 await page.getByRole('button',{name:'Stop',exact:true}).click();
 await page.waitForFunction(()=>document.getElementById('status').textContent.includes('Transcript complete'),{},{timeout:10000});
 const transcript=await page.locator('#transcript').inputValue();
 assert(/[\u0600-\u06ff]/.test(transcript));assert(updates.some(m=>m.type==='ack'));assert(updates.some(m=>m.type==='final'));
 assert(await page.evaluate(()=>window.testStream.getTracks().every(t=>t.readyState==='ended')));
 assert.deepEqual(errors,[]);
 if (process.env.MIC_TEST_SCREENSHOT) await page.screenshot({path:process.env.MIC_TEST_SCREENSHOT,fullPage:true});
 console.log(JSON.stringify({test:'Browser audio capture through the configured real model',transcript,updates,errors,result:'passed'},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exit(1)});
