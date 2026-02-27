import { initializeApp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-app.js";
import { getFirestore, doc, updateDoc, serverTimestamp } from "https://www.gstatic.com/firebasejs/10.12.2/firebase-firestore.js";

const firebaseConfig = {
    apiKey: "AIzaSyDv7BgLJM-xiwv1AWQJGrADRZnemWGgiUI",
    authDomain: "dementia-screening.firebaseapp.com",
    projectId: "dementia-screening",
    storageBucket: "dementia-screening.firebasestorage.app",
    messagingSenderId: "948621756070",
    appId: "1:948621756070:web:dc2a7bcc43c1c3557a1667",
    measurementId: "G-YPGKV8W0QH"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

window.saveScoresToFirebase = async function (scoreData) {
    const uid = sessionStorage.getItem('nb_uid');
    if (!uid) {
        console.warn('[Firebase] No user logged in. Scores will not be saved to Firebase.');
        return;
    }

    try {
        const updates = {};
        if (scoreData.memory !== undefined) updates['scores.memory'] = scoreData.memory;
        if (scoreData.nback !== undefined) updates['scores.nback'] = scoreData.nback;
        if (scoreData.questionnaire !== undefined) updates['scores.questionnaire'] = scoreData.questionnaire;
        if (scoreData.mlPrediction !== undefined) updates['scores.mlPrediction'] = scoreData.mlPrediction;
        updates['scores.lastUpdated'] = serverTimestamp();

        await updateDoc(doc(db, 'users', uid), updates);
        console.log('[Firebase] Scores successfully saved to user database:', updates);
    } catch (err) {
        console.error('[Firebase] Could not save scores:', err.message);
    }
};
