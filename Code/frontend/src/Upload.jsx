import { useRef } from "react"

function Upload({ onTranscribed }) {
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
                .then(res => res.json())
                .then(data => {
                    onTranscribed({
                        midiUrl: `http://localhost:5000${data.midiUrl}?t=${Date.now()}`,
                        audioUrl: `http://localhost:5000${data.audioUrl}?t=${Date.now()}`
                    })
                    event.target.value = ""
                })
            }}/>
            
            <button onClick={() => fileInputRef.current.click()}>Upload</button>
        </div>
    )
}

export default Upload