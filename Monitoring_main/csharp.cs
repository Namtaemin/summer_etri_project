using System;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Threading.Tasks;
using Newtonsoft.Json.Linq;
using System.Text;

namespace vmpoStudy01
{
    class Program
    {

        static void Main(string[] args)
        {
            String callUrl = "http://192.168.0.2:5000/getValue";
    
            String postData = "";
        
            HttpWebRequest httpWebRequest = (HttpWebRequest) WebRequest.Create(callUrl);
            // 인코딩 UTF-8
            byte[] sendData = Encoding.UTF8.GetBytes(postData);
            httpWebRequest.ContentType = "application/x-www-form-urlencoded; charset=UTF-8";
            // 전송 타입 지정
            httpWebRequest.Method = "POST";
            // 전송 길이 
            httpWebRequest.ContentLength = sendData.Length;
            // 요청 스트림 활성화
            Stream requestStream = httpWebRequest.GetRequestStream();
            requestStream.Write(sendData, 0, sendData.Length);
            requestStream.Close();
            // 수신 내용
            HttpWebResponse httpWebResponse = (HttpWebResponse) httpWebRequest.GetResponse();
            StreamReader streamReader = new StreamReader(httpWebResponse.GetResponseStream(), Encoding.GetEncoding("UTF-8")); 
            // 스트림 끝까지 읽기   
            string result = streamReader.ReadToEnd();
            streamReader.Close();
            httpWebResponse.Close();
            Console.Write("return: " + result);
        }
    }
}
