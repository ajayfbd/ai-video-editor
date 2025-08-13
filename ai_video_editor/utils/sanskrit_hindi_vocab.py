"""
Comprehensive Sanskrit and Hindi vocabulary system for better ASR recognition.
Contains extensive word lists for religious, classical, and common terms.
"""

from typing import List, Dict, Set
import random

class SanskritHindiVocabulary:
    """Comprehensive vocabulary system for Sanskrit and Hindi ASR enhancement."""
    
    def __init__(self):
        self._religious_terms = self._get_religious_terms()
        self._classical_terms = self._get_classical_terms()
        self._common_hindi = self._get_common_hindi()
        self._deity_names = self._get_deity_names()
        self._scripture_terms = self._get_scripture_terms()
        self._philosophical_terms = self._get_philosophical_terms()
        self._ritual_terms = self._get_ritual_terms()
        self._mythological_terms = self._get_mythological_terms()
    
    def _get_religious_terms(self) -> List[str]:
        """Core religious and devotional terms."""
        return [
            # Core devotional terms
            "भगवान", "प्रभु", "ईश्वर", "परमेश्वर", "भगवत्", "देव", "देवी", "दिव्य",
            "पूज्य", "श्री", "महा", "परम", "सत्", "चित्", "आनंद", "ब्रह्म",
            
            # Worship and devotion
            "भक्ति", "प्रेम", "श्रद्धा", "विश्वास", "समर्पण", "सेवा", "पूजा", "अर्चना",
            "वंदना", "स्तुति", "स्तोत्र", "प्रार्थना", "आराधना", "उपासना", "ध्यान", "जप",
            
            # Blessings and grace
            "आशीर्वाद", "कृपा", "अनुग्रह", "प्रसाद", "दर्शन", "चरण", "शरण", "रक्षा",
            "कल्याण", "मंगल", "शुभ", "लाभ", "सिद्धि", "सफलता", "विजय", "जय",
            
            # Spiritual concepts
            "आत्मा", "परमात्मा", "जीव", "चेतना", "चैतन्य", "शक्ति", "तेज", "प्रकाश",
            "ज्योति", "दिव्यता", "पवित्रता", "शुद्धता", "सत्यता", "धर्म", "सत्य", "अहिंसा"
        ]
    
    def _get_deity_names(self) -> List[str]:
        """Names of major deities and divine figures."""
        return [
            # Vishnu and avatars
            "विष्णु", "नारायण", "हरि", "केशव", "माधव", "गोविंद", "दामोदर", "वासुदेव",
            "राम", "रामचंद्र", "सीताराम", "रघुनाथ", "रघुपति", "कोसलेश", "दशरथि",
            "कृष्ण", "कन्हैया", "गोपाल", "मुरारी", "मोहन", "श्याम", "बांके", "बिहारी",
            "नरसिंह", "नृसिंह", "प्रह्लाद", "हिरण्यकशिपु", "वराह", "कूर्म", "मत्स्य",
            "वामन", "परशुराम", "बुद्ध", "कल्कि", "जगन्नाथ", "वेंकटेश", "तिरुपति",
            
            # Shiva and family
            "शिव", "शंकर", "महादेव", "भोलेनाथ", "नीलकंठ", "त्रिलोचन", "गंगाधर",
            "नटराज", "दक्षिणामूर्ति", "अर्धनारीश्वर", "सदाशिव", "ईशान", "रुद्र",
            "पार्वती", "उमा", "गौरी", "दुर्गा", "काली", "चंडी", "अंबिका", "भवानी",
            "गणेश", "गणपति", "विनायक", "लंबोदर", "एकदंत", "वक्रतुंड", "विघ्नेश",
            "कार्तिकेय", "स्कंद", "मुरुगन", "सुब्रह्मण्य", "षण्मुख", "कुमार",
            
            # Devi forms
            "लक्ष्मी", "श्री", "महालक्ष्मी", "धनलक्ष्मी", "ऐश्वर्य", "संपदा",
            "सरस्वती", "वीणापाणि", "शारदा", "वागीश्वरी", "ब्रह्माणी", "विद्या",
            "राधा", "राधिका", "वृषभानुजा", "कृष्णप्रिया", "गोपी", "ब्रजरानी",
            "सीता", "जानकी", "मैथिली", "वैदेही", "भूमिजा", "रामप्रिया",
            
            # Other major deities
            "ब्रह्मा", "सरस्वती", "इंद्र", "अग्नि", "वायु", "वरुण", "यम", "कुबेर",
            "सूर्य", "चंद्र", "मंगल", "बुध", "गुरु", "शुक्र", "शनि", "राहु", "केतु",
            "हनुमान", "मारुति", "पवनपुत्र", "बजरंगबली", "संकटमोचन", "रामभक्त"
        ]
    
    def _get_classical_terms(self) -> List[str]:
        """Classical Sanskrit and philosophical terms."""
        return [
            # Philosophical concepts
            "दर्शन", "तत्त्व", "सिद्धांत", "मत", "वाद", "न्याय", "तर्क", "युक्ति",
            "प्रमाण", "अनुमान", "प्रत्यक्ष", "शब्द", "उपमान", "अर्थापत्ति", "अनुपलब्धि",
            "सांख्य", "योग", "न्याय", "वैशेषिक", "मीमांसा", "वेदांत", "चार्वाक",
            
            # Yoga and meditation
            "योग", "ध्यान", "धारणा", "समाधि", "प्राणायाम", "आसन", "यम", "नियम",
            "प्रत्याहार", "एकाग्रता", "वैराग्य", "अभ्यास", "समस्कार", "वृत्ति",
            "चित्त", "मन", "बुद्धि", "अहंकार", "इंद्रिय", "विषय", "संयम", "कैवल्य",
            
            # Spiritual states and goals
            "मोक्ष", "मुक्ति", "निर्वाण", "कैवल्य", "सिद्धि", "समाधि", "तुरीय",
            "सच्चिदानंद", "आत्मसाक्षात्कार", "ब्रह्मज्ञान", "तत्त्वज्ञान", "स्वरूप",
            "साक्षी", "द्रष्टा", "चैतन्य", "चित्", "सत्", "आनंद", "शांति", "प्रेम",
            
            # Ayurveda and health
            "आयुर्वेद", "वैद्य", "चिकित्सा", "औषधि", "रसायन", "दोष", "वात", "पित्त",
            "कफ", "धातु", "मल", "ओजस", "तेजस", "प्राण", "अपान", "समान", "उदान", "व्यान",
            "चक्र", "नाडी", "कुंडलिनी", "सुषुम्ना", "इडा", "पिंगला", "मूलाधार", "स्वाधिष्ठान"
        ]
    
    def _get_scripture_terms(self) -> List[str]:
        """Names of scriptures and religious texts."""
        return [
            # Vedas and Upanishads
            "वेद", "ऋग्वेद", "सामवेद", "यजुर्वेद", "अथर्ववेद", "संहिता", "ब्राह्मण",
            "आरण्यक", "उपनिषद", "ईशावास्य", "केन", "कठ", "प्रश्न", "मुंडक", "मांडूक्य",
            "तैत्तिरीय", "ऐतरेय", "छांदोग्य", "बृहदारण्यक", "श्वेताश्वतर", "कौशीतकी",
            
            # Epics and Puranas
            "रामायण", "महाभारत", "गीता", "भगवद्गीता", "श्रीमद्भागवत", "विष्णुपुराण",
            "शिवपुराण", "देवीभागवत", "स्कंदपुराण", "गरुड़पुराण", "ब्रह्मपुराण",
            "मत्स्यपुराण", "कूर्मपुराण", "वराहपुराण", "वामनपुराण", "नारदपुराण",
            
            # Other texts
            "स्मृति", "धर्मशास्त्र", "मनुस्मृति", "याज्ञवल्क्यस्मृति", "पराशरस्मृति",
            "तंत्र", "आगम", "संहिता", "कल्पसूत्र", "गृह्यसूत्र", "श्रौतसूत्र",
            "दर्शनशास्त्र", "व्याकरण", "छंद", "ज्योतिष", "निरुक्त", "शिक्षा"
        ]
    
    def _get_ritual_terms(self) -> List[str]:
        """Ritual and ceremonial terms."""
        return [
            # Worship items and rituals
            "पूजा", "अर्चना", "आरती", "हवन", "यज्ञ", "होम", "अभिषेक", "रुद्राभिषेक",
            "चंदन", "कुमकुम", "सिंदूर", "तिलक", "भस्म", "विभूति", "गंगाजल", "तुलसी",
            "बिल्वपत्र", "दूर्वा", "कुश", "तिल", "अक्षत", "पुष्प", "माला", "धूप",
            "दीप", "कपूर", "नैवेद्य", "प्रसाद", "चरणामृत", "पंचामृत", "घी", "दूध",
            
            # Festivals and occasions
            "त्योहार", "उत्सव", "पर्व", "व्रत", "एकादशी", "पूर्णिमा", "अमावस्या",
            "संक्रांति", "दीवाली", "होली", "दशहरा", "नवरात्रि", "जन्माष्टमी",
            "रामनवमी", "शिवरात्रि", "गणेशचतुर्थी", "करवाचौथ", "तीज", "रक्षाबंधन",
            
            # Sacred places and objects
            "मंदिर", "देवालय", "गर्भगृह", "प्रांगण", "द्वार", "शिखर", "कलश", "ध्वजा",
            "घंटा", "शंख", "दमरू", "त्रिशूल", "चक्र", "गदा", "पद्म", "कमल",
            "यंत्र", "मंत्र", "बीज", "तंत्र", "रुद्राक्ष", "माला", "जप", "स्तोत्र"
        ]
    
    def _get_mythological_terms(self) -> List[str]:
        """Mythological characters and concepts.""" 
        return [
            # Epic characters
            "प्रह्लाद", "हिरण्यकशिपु", "होलिका", "भक्त", "असुर", "दैत्य", "दानव", "राक्षस",
            "यक्ष", "गंधर्व", "अप्सरा", "किन्नर", "नाग", "गरुड़", "हंस", "मयूर",
            
            # Ramayana characters
            "दशरथ", "कौशल्या", "सुमित्रा", "कैकेयी", "भरत", "लक्ष्मण", "शत्रुघ्न",
            "सीता", "रावण", "मेघनाद", "कुंभकर्ण", "विभीषण", "सुग्रीव", "बाली",
            "अंगद", "जामवंत", "नल", "नील", "जटायु", "सम्पाती", "अहल्या",
            
            # Mahabharata characters
            "युधिष्ठिर", "भीम", "अर्जुन", "नकुल", "सहदेव", "द्रौपदी", "कुंती",
            "दुर्योधन", "दुःशासन", "शकुनि", "कर्ण", "द्रोण", "भीष्म", "विदुर",
            "धृतराष्ट्र", "गांधारी", "अभिमन्यु", "घटोत्कच", "बर्बरीक", "एकलव्य",
            
            # Krishna stories
            "यशोदा", "नंद", "गोकुल", "वृंदावन", "मथुरा", "द्वारका", "कंस", "पूतना",
            "त्रिणावर्त", "बकासुर", "अघासुर", "धेनुकासुर", "प्रलंबासुर", "अरिष्टासुर",
            "केशी", "व्योमासुर", "गोवर्धन", "कालिया", "इंद्र", "वरुण", "कुब्जा"
        ]
    
    def _get_common_hindi(self) -> List[str]:
        """Common Hindi words that might be mispronounced."""
        return [
            # Common verbs
            "करना", "होना", "जाना", "आना", "देना", "लेना", "कहना", "सुनना", "देखना",
            "समझना", "पढ़ना", "लिखना", "खाना", "पीना", "सोना", "उठना", "बैठना",
            "चलना", "दौड़ना", "रुकना", "ठहरना", "मिलना", "बिछड़ना", "हंसना", "रोना",
            
            # Common nouns
            "घर", "परिवार", "माता", "पिता", "भाई", "बहन", "पुत्र", "पुत्री", "पति",
            "पत्नी", "दादा", "दादी", "नाना", "नानी", "चाचा", "चाची", "मामा", "मामी",
            "फूफा", "बुआ", "ससुर", "सास", "जेठ", "देवर", "ननद", "भाभी", "साला",
            
            # Time and numbers
            "समय", "दिन", "रात", "सुबह", "दोपहर", "शाम", "सप्ताह", "महीना", "साल",
            "एक", "दो", "तीन", "चार", "पांच", "छह", "सात", "आठ", "नौ", "दस",
            "बीस", "तीस", "चालीस", "पचास", "साठ", "सत्तर", "अस्सी", "नब्बे", "सौ"
        ]
    
    def _get_philosophical_terms(self) -> List[str]:
        """Advanced philosophical and spiritual terms."""
        return [
            # Consciousness and reality
            "चेतना", "चैतन्य", "साक्षी", "द्रष्टा", "दृश्य", "माया", "लीला", "स्वप्न",
            "जाग्रत", "सुषुप्ति", "तुरीय", "अवस्था", "कोश", "शरीर", "स्थूल", "सूक्ष्म",
            "कारण", "उपाधि", "अध्यास", "अविद्या", "विद्या", "ज्ञान", "अज्ञान", "भ्रम",
            
            # Vedantic concepts
            "अद्वैत", "द्वैत", "विशिष्टाद्वैत", "द्वैताद्वैत", "शुद्धाद्वैत", "अचिंत्यभेदाभेद",
            "ब्रह्म", "आत्मन्", "जीव", "ईश्वर", "प्रकृति", "पुरुष", "गुण", "सत्त्व",
            "रजस्", "तमस्", "त्रिगुण", "निर्गुण", "सगुण", "निराकार", "साकार",
            
            # Karma and liberation
            "कर्म", "विकर्म", "अकर्म", "संस्कार", "वासना", "फल", "बंधन", "मुक्ति",
            "जन्म", "मृत्यु", "पुनर्जन्म", "संसार", "भव", "भवसागर", "तार", "पार"
        ]
    
    def get_vocabulary_prompt(self, category: str = "comprehensive", max_words: int = 100) -> str:
        """
        Generate vocabulary prompt for ASR initial_prompt.
        
        Args:
            category: Type of vocabulary ("religious", "classical", "comprehensive", "mythological")
            max_words: Maximum number of words to include
            
        Returns:
            Comma-separated string of vocabulary words
        """
        if category == "religious":
            words = self._religious_terms + self._deity_names + self._ritual_terms
        elif category == "classical":
            words = self._classical_terms + self._philosophical_terms + self._scripture_terms
        elif category == "mythological":
            words = self._mythological_terms + self._deity_names
        elif category == "comprehensive":
            words = (self._religious_terms + self._deity_names + self._classical_terms + 
                    self._scripture_terms + self._ritual_terms + self._mythological_terms +
                    self._philosophical_terms)
        else:
            words = self._common_hindi
        
        # Remove duplicates and shuffle for variety
        unique_words = list(set(words))
        random.shuffle(unique_words)
        
        # Limit to max_words
        selected_words = unique_words[:max_words]
        
        return ", ".join(selected_words)
    
    def get_contextual_vocabulary(self, detected_terms: List[str], max_words: int = 50) -> str:
        """
        Get contextually relevant vocabulary based on detected terms.
        
        Args:
            detected_terms: List of terms already detected in the audio
            max_words: Maximum words to return
            
        Returns:
            Contextually relevant vocabulary prompt
        """
        relevant_words = set()
        
        # Check for religious context
        religious_indicators = {"भगवान", "प्रभु", "देव", "पूजा", "मंत्र", "आरती", "भजन"}
        if any(term in detected_terms for term in religious_indicators):
            relevant_words.update(self._religious_terms[:20])
            relevant_words.update(self._deity_names[:15])
            relevant_words.update(self._ritual_terms[:15])
        
        # Check for mythological context
        mythological_indicators = {"राम", "कृष्ण", "शिव", "रामायण", "महाभारत", "प्रह्लाद", "हिरण्यकशिपु"}
        if any(term in detected_terms for term in mythological_indicators):
            relevant_words.update(self._mythological_terms[:25])
        
        # Check for philosophical context
        philosophical_indicators = {"योग", "ध्यान", "आत्मा", "ब्रह्म", "मोक्ष", "गीता"}
        if any(term in detected_terms for term in philosophical_indicators):
            relevant_words.update(self._philosophical_terms[:20])
            relevant_words.update(self._classical_terms[:15])
        
        # If no specific context, use general religious vocabulary
        if not relevant_words:
            relevant_words.update(self._religious_terms[:30])
            relevant_words.update(self._common_hindi[:20])
        
        # Convert to list and limit
        word_list = list(relevant_words)[:max_words]
        return ", ".join(word_list)
    
    def get_all_categories(self) -> Dict[str, List[str]]:
        """Get all vocabulary categories."""
        return {
            "religious": self._religious_terms,
            "deities": self._deity_names,
            "classical": self._classical_terms,
            "scriptures": self._scripture_terms,
            "rituals": self._ritual_terms,
            "mythological": self._mythological_terms,
            "philosophical": self._philosophical_terms,
            "common_hindi": self._common_hindi
        }
    
    def get_vocabulary_stats(self) -> Dict[str, int]:
        """Get statistics about vocabulary size."""
        categories = self.get_all_categories()
        stats = {category: len(words) for category, words in categories.items()}
        stats["total_unique"] = len(set().union(*categories.values()))
        return stats

# Global instance for easy access
sanskrit_hindi_vocab = SanskritHindiVocabulary()