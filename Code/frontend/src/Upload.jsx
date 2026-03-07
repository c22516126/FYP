import { useState, useRef } from "react"

function Upload() {
    const [audioUrl, setAudioUrl] = useState(null)
    const fileInputRef = useRef(null)

    return(
        <div id = "uploadContainer">
            <input 
                type="file"
                accept="audio/*"
                ref={fileInputRef}
                style={{display: "none"}}
                onChange={(event) => {
                    
                const inputFile = new FormData()
                const file = event.target.files[0]
                inputFile.append('file', file)

                fetch('http://localhost:5000/', {
                    method: "POST",
                    body: inputFile
                })
                .then(res => res.blob())
                .then(blob => {
                    const url = URL.createObjectURL(blob)
                    setAudioUrl(url)
                    event.target.value = null
                })
            }}/>
            
            <button onClick={() => fileInputRef.current.click()}>Upload</button>
            {audioUrl && (
                <audio key={audioUrl} controls>
                    <source src={audioUrl} type="audio/wav"/>
                    Audio not supported by browser.
                </audio>
            )}
        </div>
    )
}

export default Upload