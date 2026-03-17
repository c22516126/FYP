import { useState } from "react"
import Upload from "./Upload.jsx"
import MidiPlayer from "./MidiPlayer.jsx"
function Body() {

    const [urls, setUrls] = useState(null)

    return (
        <>
            <div id = "bodyContent">
                <p>Insert your own audio file. The song will be returned as a piano roll with audio, showing you how the song is composed.</p>
                <Upload onTranscribed={setUrls} />
                {urls && <MidiPlayer midiUrl={urls.midiUrl} audioUrl={urls.audioUrl} />}
            </div>
        </>
    )
}


export default Body