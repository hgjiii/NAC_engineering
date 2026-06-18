import requests
from Bio import Entrez, SeqIO
from io import StringIO


def configure_entrez(email):
    """
    Description:
        Set the contact email NCBI Entrez requires for efetch requests.

    Args:
        email: Contact email address registered with the Entrez calls.
    """
    Entrez.email = email


def _fetch_uniprotkb(accession):
    """Return UniProtKB JSON if entry is active and has sequence, else None."""
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if "sequence" not in data:
        return None
    return data


def _fetch_uniparc_ebi(accession):
    """Return UniParc record from EBI Proteins API (works for obsolete entries)."""
    url = f"https://www.ebi.ac.uk/proteins/api/uniparc/accession/{accession}"
    resp = requests.get(url, headers={"Accept": "application/json"}, timeout=30)
    if resp.status_code != 200:
        return None
    data = resp.json()
    if isinstance(data, list):
        return data[0] if data else None
    return data


def _get_property(xref, key):
    """Helper: extract property value by type from a UniParc dbReference."""
    for p in xref.get("property", []):
        if p.get("type") == key:
            return p.get("value")
    return None


def get_sequence(accession):
    """
    Description:
        Fetch protein sequence; falls back to UniParc (EBI) if UniProtKB lacks it.
    """
    data = _fetch_uniprotkb(accession)
    if data:
        return data["sequence"]["value"]

    archive = _fetch_uniparc_ebi(accession)
    if archive and "sequence" in archive:
        seq = archive["sequence"]
        if isinstance(seq, dict):
            result = seq.get("content") or seq.get("value")
        else:
            result = seq
        if result:
            return result

    print(f"[None] sequence: {accession}")
    return None


def get_taxonomy(accession):
    """
    Description:
        Fetch organism info; falls back to UniParc cross-references if obsolete.
    """
    data = _fetch_uniprotkb(accession)
    if data and "organism" in data:
        org = data["organism"]
        return {
            "accession": accession,
            "tax_id": org["taxonId"],
            "scientific_name": org["scientificName"],
            "common_name": org.get("commonName"),
        }

    archive = _fetch_uniparc_ebi(accession)
    if archive:
        for xref in archive.get("dbReference", []):
            if xref.get("active") != "Y":
                continue
            tax_id = _get_property(xref, "NCBI_taxonomy_id")
            if tax_id:
                return {
                    "accession": accession,
                    "tax_id": int(tax_id),
                    "scientific_name": None,
                    "common_name": None,
                }

    print(f"[None] taxonomy: {accession}")
    return {"accession": accession, "scientific_name": None,
            "tax_id": None, "common_name": None}


def get_dna_from_uniprot(uniprot_accession):
    """
    Description:
        Fetch CDS DNA via EMBL xref; falls back to UniParc (EBI) if obsolete.
    """
    data = _fetch_uniprotkb(uniprot_accession)
    embl_xrefs = []
    if data:
        embl_xrefs = [x for x in data.get("uniProtKBCrossReferences", [])
                      if x["database"] == "EMBL"]

    if not embl_xrefs:
        archive = _fetch_uniparc_ebi(uniprot_accession)
        if archive:
            for xref in archive.get("dbReference", []):
                if xref.get("type") not in ("EMBL", "EMBLWGS"):
                    continue
                if xref.get("active") != "Y":
                    continue
                embl_xrefs.append({
                    "id": xref.get("id"),
                    "properties": [{"key": "ProteinId", "value": xref.get("id")}],
                })
    if not embl_xrefs:
        print(f"[None] dna: {uniprot_accession}")
        return None

    embl_id = embl_xrefs[0]["id"]
    protein_id = None
    for prop in embl_xrefs[0].get("properties", []):
        if prop["key"] == "ProteinId":
            protein_id = prop["value"]
            break

    if protein_id and protein_id != "-":
        handle = Entrez.efetch(db="protein", id=protein_id,
                               rettype="fasta_cds_na", retmode="text")
        fasta_text = handle.read()
        handle.close()
        record = next(SeqIO.parse(StringIO(fasta_text), "fasta"))
        dna_seq = str(record.seq)
    else:
        handle = Entrez.efetch(db="nucleotide", id=embl_id,
                               rettype="fasta", retmode="text")
        record = next(SeqIO.parse(handle, "fasta"))
        handle.close()
        dna_seq = str(record.seq)

    return {"embl_id": embl_id, "protein_id": protein_id, "dna": dna_seq}
