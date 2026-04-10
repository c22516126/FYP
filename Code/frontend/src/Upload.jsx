import { useRef, useState } from "react"

function Upload({ onTranscribed }) {

    const fileInputRef = useRef(null)
    const [loading, setLoading] = useState(false)

    return(
        <div id = "uploadContainer">
            <input 
                type="file"
                accept="audio/*"
                ref={fileInputRef}
                style={{display: "none"}}
                onChange={(event) => {

                    const file = event.target.files[0]
                    if (!file) return

                    setLoading(true) 
                    
                    const inputFile = new FormData()
                    inputFile.append('file', file)

                    fetch('http://localhost:5000/', {
                        method: "POST",
                        body: inputFile
                    })
                    .then(res => res.json())
                    .then(data => {
                        onTranscribed({
                            midiUrl: `http://localhost:5000${data.midiUrl}?t=${Date.now()}`
                        })
                        event.target.value = ""
                    })
                    .catch(err => {
                        console.error(err)
                    })
                    .finally(() => {
                        setLoading(false) 
                        event.target.value = ""
                    })

                }}/>
            
            <button 
                onClick={() => fileInputRef.current.click()}
                disabled={loading}
            >
                {loading ? "Processing..." : "Upload"}
            </button>

            {loading && <p>Transcribing audio... this may take a few seconds</p>}
        </div>
    )
}

export default Upload