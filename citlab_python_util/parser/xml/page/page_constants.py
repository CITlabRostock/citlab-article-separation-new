# Creators name
sCREATOR = "CITlab"

# Namespace for PageXml
NS_PAGE_XML = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"

NS_XSI = "http://www.w3.org/2001/XMLSchema-instance"
XSILOCATION = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 " \
              "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd"

# Schema for Transkribus PageXml
XSL_SCHEMA_FILENAME = "pagecontent_transkribus.xsd"

# XML schema loaded once for all
cachedValidationContext = None

sMETADATA_ELT = "Metadata"
sCREATOR_ELT = "Creator"
sCREATED_ELT = "Created"
sLAST_CHANGE_ELT = "LastChange"
sCOMMENTS_ELT = "Comments"
sTranskribusMetadata_ELT = "TranskribusMetadata"
sPRINT_SPACE = "PrintSpace"
sCUSTOM_ATTR = "custom"
sTEXTLINE = "TextLine"
sBASELINE = "Baseline"
sWORD = "Word"
sCOORDS = "Coords"
sTEXTEQUIV = "TextEquiv"
sUNICODE = "Unicode"

sPOINTS_ATTR = "points"
sREADING_ORDER = "readingOrder"

sTEXTREGION = "TextRegion"
sIMAGEREGION = "ImageRegion"
sLINEDRAWINGREGION = "LineDrawingRegion"
sGRAPHICREGION = "GraphicRegion"
sTABLEREGION = "TableRegion"
sCHARTREGION = "ChartRegion"
sSEPARATORREGION = "SeparatorRegion"
sMATHSREGION = "MathsRegion"
sCHEMREGION = "ChemRegion"
sMUSICREGION = "MusicRegion"
sADVERTREGION = "AdvertRegion"
sNOISEREGION = "NoiseRegion"
sUNKNOWNREGION = "UnknownRegion"

sEXT = ".xml"


# TextRegion Types
class TextRegionTypes:
    sPARAGRAPH = "paragraph"
    sHEADING = "heading"
    sCAPTION = "caption"
    sHEADER = "header"
    sFOOTER = "footer"
    sPAGENUMBER = "page-number"
    sDROPCAPITAL = "drop-capital"
    sCREDIT = "credit"
    sFLOATING = "floating"
    sSIGNATUREMARK = "signature-mark"
    sCATCHWORD = "catch-word"
    sMARGINALIA = "marginalia"
    sFOOTNOTE = "footnote"
    sFOOTNOTECONT = "footnote-continued"
    sENDNOTE = "endnote"
    sTOCENTRY = "TOC-entry"
    sOTHER = "other"
