using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
public class buttonEvent1 : MonoBehaviour
{
    public GameObject particle;
    public Button btn;
    public Text t;
    // Start is called before the first frame update
    void Start()
    {
        btn.onClick.AddListener(btnprint);
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void btnprint()
    {
        particle.SetActive(!particle.activeSelf);
        if(t.GetComponent<Text>().text == "제습기 ON/OFF" || t.GetComponent<Text>().text == "제습기 ON")
        {    
            t.GetComponent<Text>().text = "제습기 OFF";
        }
        else
        {
            t.GetComponent<Text>().text = "제습기 ON";
        }
    }
}
