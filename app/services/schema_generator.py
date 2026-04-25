import json
from datetime import datetime, timezone


def generate_article_schema(
    article_title: str,
    article_description: str,
    article_url: str,
    target_keyword: str,
    business_name: str,
    business_category: str | None = None,
    business_location: str | None = None,
    business_rating: float | None = None,
    business_review_count: int | None = None,
    published_at: str | None = None,
) -> list[dict]:
    schemas = []

    now = published_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    schemas.append({
        "@context": "https://schema.org",
        "@type": "Article",
        "headline": article_title,
        "description": article_description,
        "keywords": target_keyword,
        "author": {
            "@type": "Organization",
            "name": business_name,
        },
        "publisher": {
            "@type": "Organization",
            "name": business_name,
        },
        "datePublished": now,
        "dateModified": now,
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": article_url,
        },
    })

    local_biz: dict = {
        "@context": "https://schema.org",
        "@type": "LocalBusiness",
        "name": business_name,
    }
    if business_category:
        CATEGORY_MAP = {
            "salon": "HairSalon",
            "barbershop": "BarberShop",
            "barber": "BarberShop",
            "restaurant": "Restaurant",
            "cafe": "CafeOrCoffeeShop",
            "bakery": "Bakery",
            "gym": "ExerciseGym",
            "dentist": "Dentist",
            "doctor": "Physician",
            "hotel": "Hotel",
            "spa": "DaySpa",
            "store": "Store",
            "pharmacy": "Pharmacy",
            "bar": "BarOrPub",
            "lawyer": "LegalService",
            "plumber": "Plumber",
            "electrician": "Electrician",
            "real estate": "RealEstateAgent",
        }
        schema_type = CATEGORY_MAP.get(business_category.lower(), "LocalBusiness")
        local_biz["@type"] = schema_type

    if business_location:
        local_biz["address"] = {
            "@type": "PostalAddress",
            "streetAddress": business_location,
        }

    if business_rating is not None:
        local_biz["aggregateRating"] = {
            "@type": "AggregateRating",
            "ratingValue": str(round(business_rating, 1)),
            "bestRating": "5",
            "ratingCount": str(business_review_count or 0),
        }

    schemas.append(local_biz)

    return schemas


def generate_faq_schema(faqs: list[dict]) -> dict | None:
    if not faqs:
        return None
    return {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": [
            {
                "@type": "Question",
                "name": faq["question"],
                "acceptedAnswer": {
                    "@type": "Answer",
                    "text": faq["answer"],
                },
            }
            for faq in faqs
        ],
    }


def schemas_to_jsonld(schemas: list[dict]) -> str:
    return "\n".join(
        f'<script type="application/ld+json">{json.dumps(s, ensure_ascii=False)}</script>'
        for s in schemas
        if s
    )
